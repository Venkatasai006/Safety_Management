from flask import Flask, render_template, Response
import cv2
import threading
import time
import winsound
from playsound import playsound
from twilio.rest import Client
from ultralytics import YOLO
import os
import requests

app = Flask(__name__)

# Twilio Credentials
TWILIO_WHATSAPP = "+14155238886"
SUPERVISOR_WHATSAPP = "+918309872933"
account_sid = 'AC7dd1a44883af0e8f7400b15f3d9d314c'
auth_token = '4321d1dc16b9bacfc074240de7afe013'
client = Client(account_sid, auth_token)

# Load YOLO Model
model = YOLO(r"C:\Users\venka\Downloads\best.pt")

# Class Names
CLASS_NAMES = {0: "Helmet", 2: "No Helmet", 7: "Vest", 4: "No Vest"}

# Alarm Sound Path
ALARM_SOUND = r"C:\Users\venka\Downloads\alarm.mp3"

# Create screenshots directory
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

last_alert_time = 0
violation_detected_time = None

def upload_image(image_path):
    api_key = "62f092d8d8f6d354f613bbec5dff4c55"
    with open(image_path, "rb") as file:
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": api_key},
            files={"image": file}
        )
    if response.status_code == 200:
        return response.json()["data"]["url"]
    else:
        print("Error uploading image:", response.text)
        return None

def send_alert_with_screenshot(image_path):
    image_url = upload_image(image_path)
    if image_url:
        try:
            message = client.messages.create(
                from_='whatsapp:' + TWILIO_WHATSAPP,
                body="🚨 ALERT: Worker detected without safety gear! ⚠",
                to='whatsapp:' + SUPERVISOR_WHATSAPP,
                media_url=[image_url]
            )
            print("WhatsApp Alert Sent!")
        except Exception as e:
            print("Error sending WhatsApp alert:", e)

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 5)
    global last_alert_time, violation_detected_time
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame, stream=True)
        violations_detected = False
        
        # Count classes
        counts = {"Helmet": 0, "No Helmet": 0, "Vest": 0, "No Vest": 0}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0].item())
                class_name = CLASS_NAMES.get(class_id, "Unknown")
                
                if class_name in counts:
                    counts[class_name] += 1
                
                if class_name in ["No Helmet", "No Vest"]:
                    violations_detected = True
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif class_name in ["Helmet", "Vest"]:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display counts on frame
        y_offset = 20
        for key, value in counts.items():
            cv2.putText(frame, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 20

        if violations_detected:
            if violation_detected_time is None:
                violation_detected_time = time.time()
        else:
            violation_detected_time = None
        
        if violation_detected_time and (time.time() - violation_detected_time >= 2):
            violation_detected_time = None
            try:
                winsound.Beep(1000, 500)
                playsound(ALARM_SOUND)
            except Exception as e:
                print("Error playing alarm:", e)
            
            screenshot_path = f"screenshots/violation_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
            
            if time.time() - last_alert_time > 10:
                last_alert_time = time.time()
                threading.Thread(target=send_alert_with_screenshot, args=(screenshot_path,), daemon=True).start()
        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)