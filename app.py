from flask import Flask, render_template, Response, jsonify, url_for, request
import cv2
import mediapipe as mp
import time
import math
import threading
import queue
import os

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cart = {"Burger": [], "Pizza": [], "Pasta": [], "Thepla": [], "FrenchFries": [], "CocaCola": []}
last_gesture_time = 0  
last_detected_gesture = None  
last_added_item = None
is_modal_open = False
modal_open_time = 0
cap = None
frame_queue = queue.Queue(maxsize=1)
camera_running = False
camera_lock = threading.Lock()

def initialize_camera():
    global cap, camera_running
    with camera_lock:
        if cap is not None and cap.isOpened():
            cap.release()
        for index in range(2): 
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"Camera initialized on index {index}")
                camera_running = True
                return True
            cap.release()
        print("Failed to initialize camera")
        camera_running = False
        return False

def capture_frames():
    global cap, last_gesture_time, last_detected_gesture, last_added_item, is_modal_open, modal_open_time, camera_running, cart
    while True:
        with camera_lock:
            if not cap or not cap.isOpened():
                if not initialize_camera():
                    print("Camera reinitialization failed in capture thread")
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, reinitializing camera")
                cap.release()
                if not initialize_camera():
                    print("Camera reinitialization failed after read failure")
                    time.sleep(1)
                    continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_time = time.time()
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
            
            if current_time - last_gesture_time > 2:
                hand_landmarks = results.multi_hand_landmarks[0]
                if is_cross_fingers(hand_landmarks):
                    last_detected_gesture = "cross_fingers"
                elif is_thumb_up(hand_landmarks):
                    last_detected_gesture = "thumbs_up"
                elif is_fist(hand_landmarks):
                    last_detected_gesture = "fist"
                elif is_ok_gesture(hand_landmarks):
                    last_detected_gesture = "ok"
                elif is_pointing_up(hand_landmarks):
                    last_detected_gesture = "pointing_up"
                elif is_peace_sign(hand_landmarks):
                    last_detected_gesture = "peace_sign"
                elif is_open_hand(hand_landmarks):
                    last_detected_gesture = "open_hand"
                elif is_first_and_last_fingers_up(hand_landmarks):
                    last_detected_gesture = "first_and_last_fingers_up"
                elif is_three_fingers_up(hand_landmarks):
                    last_detected_gesture = "three_fingers_up"
                else:
                    last_detected_gesture = None
                last_gesture_time = current_time
                print(f"Gesture detected: {last_detected_gesture}, Modal Open: {is_modal_open}, Cart: {cart}")
        else:
            last_detected_gesture = None

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            frame_bytes = buffer.tobytes()
            try:
                frame_queue.put_nowait(frame_bytes)
            except queue.Full:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame_bytes)

        time.sleep(0.033)

if not initialize_camera():
    print("Exiting due to initial webcam failure...")
    exit()
camera_thread = threading.Thread(target=capture_frames, daemon=True)
camera_thread.start()

def generate_frames():
    while True:
        try:
            frame_bytes = frame_queue.get(timeout=1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            print("Frame queue empty, waiting for camera thread")
            continue

@app.route('/')
def index():
    global last_detected_gesture
    last_detected_gesture = None
    return render_template('index.html', cart=cart)

@app.route('/order')
def order():
    global last_detected_gesture
    last_detected_gesture = None
    return render_template('order.html', cart=cart)

@app.route('/carts')
def carts():
    global last_detected_gesture
    last_detected_gesture = None
    return render_template('cart.html', cart=cart)

@app.route('/get_cart')
def get_cart():
    global cart
    cart_summary = {key: len(items) for key, items in cart.items()}
    print(f"Cart summary sent: {cart_summary}")
    return jsonify({"cart": cart, "summary": cart_summary})

@app.route('/reset_cart', methods=['POST'])
def reset_cart():
    global cart, last_added_item
    cart = {"Burger": [], "Pizza": [], "Pasta": [], "Thepla": [], "FrenchFries": [], "CocaCola": []}
    last_added_item = None
    return jsonify({"status": "success", "cart": cart})

@app.route('/remove_item', methods=['POST'])
def remove_item():
    global cart
    data = request.json
    item = data.get('item')
    customizations = data.get('customizations', [])
    
    if item in cart and len(cart[item]) > 0:
        if customizations:
            # Remove specific item with matching customizations
            for i, entry in enumerate(cart[item]):
                if sorted(entry["customizations"]) == sorted(customizations):
                    cart[item].pop(i)
                    print(f"Removed {item} with customizations: {customizations}, Cart: {cart}")
                    return jsonify({"status": "success", "cart": cart})
            return jsonify({"status": "error", "message": f"No {item} with specified customizations found"})
        else:
            # Remove first instance of the item (for gestures)
            cart[item].pop(0)
            print(f"Removed {item} without specific customizations, Cart: {cart}")
            return jsonify({"status": "success", "cart": cart})
    return jsonify({"status": "error", "message": f"No {item} in cart"})

@app.route('/add_burger', methods=['POST'])
def add_burger():
    global cart, last_added_item
    customizations = request.json.get('customizations', [])
    if not any(entry["customizations"] == customizations for entry in cart["Burger"]):
        cart["Burger"].append({"customizations": customizations})
    last_added_item = "Burger"
    print(f"Burger added via POST with customizations: {customizations}, Cart: {cart}")
    return jsonify({"status": "success", "cart": cart})

@app.route('/add_pizza', methods=['POST'])
def add_pizza():
    global cart, last_added_item
    customizations = request.json.get('customizations', [])
    if not any(entry["customizations"] == customizations for entry in cart["Pizza"]):
        cart["Pizza"].append({"customizations": customizations})
    last_added_item = "Pizza"
    print(f"Pizza added via POST with customizations: {customizations}, Cart: {cart}")
    return jsonify({"status": "success", "cart": cart})

@app.route('/add_pasta', methods=['POST'])
def add_pasta():
    global cart, last_added_item
    customizations = request.json.get('customizations', [])
    if not any(entry["customizations"] == customizations for entry in cart["Pasta"]):
        cart["Pasta"].append({"customizations": customizations})
    last_added_item = "Pasta"
    print(f"Pasta added via POST with customizations: {customizations}, Cart: {cart}")
    return jsonify({"status": "success", "cart": cart})

@app.route('/add_thepla', methods=['POST'])
def add_thepla():
    global cart, last_added_item
    customizations = request.json.get('customizations', [])
    if not any(entry["customizations"] == customizations for entry in cart["Thepla"]):
        cart["Thepla"].append({"customizations": customizations})
    last_added_item = "Thepla"
    print(f"Thepla added via POST with customizations: {customizations}, Cart: {cart}")
    return jsonify({"status": "success", "cart": cart})

@app.route('/add_frenchfries', methods=['POST'])
def add_frenchfries():
    global cart, last_added_item
    customizations = request.json.get('customizations', [])
    if not any(entry["customizations"] == customizations for entry in cart["FrenchFries"]):
        cart["FrenchFries"].append({"customizations": customizations})
    last_added_item = "FrenchFries"
    print(f"FrenchFries added via POST with customizations: {customizations}, Cart: {cart}")
    return jsonify({"status": "success", "cart": cart})

@app.route('/add_cocacola', methods=['POST'])
def add_cocacola():
    global cart, last_added_item
    customizations = request.json.get('customizations', [])
    if not any(entry["customizations"] == customizations for entry in cart["CocaCola"]):
        cart["CocaCola"].append({"customizations": customizations})
    last_added_item = "CocaCola"
    print(f"Coca-Cola added via POST with customizations: {customizations}, Cart: {cart}")
    return jsonify({"status": "success", "cart": cart})

@app.route('/check_gesture')
def check_gesture():
    global last_detected_gesture, last_added_item
    response = {"gesture": last_detected_gesture, "last_added_item": last_added_item}
    if last_added_item is not None:
        last_added_item = None
    print(f"Gesture checked: {response}")
    return jsonify(response)

@app.route('/set_modal_state', methods=['POST'])
def set_modal_state():
    global is_modal_open, modal_open_time
    is_modal_open = request.json.get('is_open', False)
    if is_modal_open:
        modal_open_time = time.time()
    print(f"Modal state set to: {is_modal_open}, Open Time: {modal_open_time}")
    return jsonify({"status": "success", "modal_open": is_modal_open})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Gesture detection functions (unchanged)
def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    thumb_extended = thumb_tip.y < thumb_ip.y and thumb_ip.y < thumb_mcp.y
    thumb_vector_x = thumb_tip.x - thumb_mcp.x
    thumb_vector_y = thumb_tip.y - thumb_mcp.y
    thumb_angle = math.degrees(math.atan2(thumb_vector_y, thumb_vector_x))
    thumb_upward = (-135 < thumb_angle < -45) or (45 < thumb_angle < 135)
    index_folded = index_tip.y > index_mcp.y
    middle_folded = middle_tip.y > middle_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    is_right_hand = thumb_tip.x < index_mcp.x
    is_left_hand = thumb_tip.x > index_mcp.x
    thumb_above_wrist = thumb_tip.y < wrist.y

    return (thumb_extended and thumb_upward and thumb_above_wrist and 
            index_folded and middle_folded and ring_folded and pinky_folded and 
            (is_right_hand or is_left_hand))

def is_fist(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    index_folded = index_tip.y > index_mcp.y + 0.02 and index_tip.y > wrist.y - 0.05
    middle_folded = middle_tip.y > middle_mcp.y + 0.02 and middle_tip.y > wrist.y - 0.05
    ring_folded = ring_tip.y > ring_mcp.y + 0.02 and ring_tip.y > wrist.y - 0.05
    pinky_folded = pinky_tip.y > pinky_mcp.y + 0.02 and pinky_tip.y > wrist.y - 0.05

    return index_folded and middle_folded and ring_folded and pinky_folded

def is_ok_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    distance_thumb_index = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    ok_circle = distance_thumb_index < 0.05
    middle_extended = middle_tip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_mcp.y

    return ok_circle and middle_extended and ring_extended and pinky_extended

def is_pointing_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    index_extended = index_tip.y < index_mcp.y
    thumb_folded = thumb_tip.y > index_mcp.y
    middle_folded = middle_tip.y > middle_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    index_above_wrist = index_tip.y < wrist.y

    return (index_extended and thumb_folded and middle_folded and ring_folded and 
            pinky_folded and index_above_wrist)

def is_peace_sign(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    thumb_folded = thumb_tip.y > index_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    fingers_above_wrist = index_tip.y < wrist.y and middle_tip.y < wrist.y
    tip_distance = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
    not_crossed = tip_distance > 0.05

    return (index_extended and middle_extended and thumb_folded and ring_folded and 
            pinky_folded and fingers_above_wrist and not_crossed)

def is_open_hand(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    thumb_extended = thumb_tip.y < index_mcp.y
    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_mcp.y
    all_above_wrist = (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
                      ring_tip.y < wrist.y and pinky_tip.y < wrist.y)

    return (thumb_extended and index_extended and middle_extended and ring_extended and 
            pinky_extended and all_above_wrist)

def is_first_and_last_fingers_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    thumb_folded = thumb_tip.y > index_mcp.y
    index_extended = index_tip.y < index_mcp.y
    middle_folded = middle_tip.y > middle_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_mcp.y
    fingers_above_wrist = index_tip.y < wrist.y and pinky_tip.y < wrist.y

    return (thumb_folded and index_extended and middle_folded and ring_folded and 
            pinky_extended and fingers_above_wrist)

def is_three_fingers_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    thumb_folded = thumb_tip.y > index_mcp.y
    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    fingers_above_wrist = (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
                          ring_tip.y < wrist.y)

    return (thumb_folded and index_extended and middle_extended and ring_extended and 
            pinky_folded and fingers_above_wrist)

def is_cross_fingers(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    fingers_crossed = (index_tip.x - middle_tip.x) * (index_pip.x - middle_pip.x) < 0
    tip_distance_y = abs(index_tip.y - middle_tip.y) < 0.03
    tip_distance_x = abs(index_tip.x - middle_tip.x) < 0.05
    thumb_folded = thumb_tip.y > index_mcp.y
    ring_folded = ring_tip.y > ring_mcp.y
    pinky_folded = pinky_tip.y > pinky_mcp.y
    fingers_above_wrist = index_tip.y < wrist.y and middle_tip.y < wrist.y

    return (index_extended and middle_extended and fingers_crossed and tip_distance_y and
            tip_distance_x and thumb_folded and ring_folded and pinky_folded and
            fingers_above_wrist)

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False, port=5005 , host='0.0.0.0' )
    finally:
        with camera_lock:
            if cap is not None and cap.isOpened():
                cap.release()
            camera_running = False
        print("Camera released on shutdown")