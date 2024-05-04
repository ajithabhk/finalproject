import cv2
from ultralytics import YOLO
import telepot
from datetime import datetime
import pytz
import numpy as np
from keras.models import load_model

# Load violence detection model
violence_model = load_model('modelnew.h5')

# Telegram Bot token and chat ID
TELEGRAM_BOT_TOKEN = []
TELEGRAM_CHAT_ID = []

# Initialize Telegram Bot
bot = telepot.Bot(token=TELEGRAM_BOT_TOKEN)

# Load YOLO model for handgun detection
model = YOLO('runs/detect/train5/weights/best.pt')




# Function to get current time
def get_current_time():
    IST = pytz.timezone('Asia/Kolkata')
    time_now = datetime.now(IST)
    return time_now


# Function to get approximate location (in this case, city name)
def get_location():
    return "PERINTHALMANNA"


# Function to perform violence detection

def detect_violence(frame):
    # Preprocess frame for violence detection (resize, normalize, etc.)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(1, 128, 128, 3) / 255.0

    # Perform violence detection
    preds = violence_model.predict(frame)
    print(f"preds_violence ={preds}")
    # Return True if violence is detected, False otherwise
    return preds[0][0] > 0.6


# Capture video from laptop camera
cap = cv2.VideoCapture(0)  # 0 indicates the default camera of your system

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform handgun detection

    results=model(frame)

    boxes = results[0].boxes  # Extract bounding boxes
    if boxes is not None:
        for det in boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().tolist()

            conf = det.conf[0].item()

            class_id = det.cls[0].item()

            if conf > 0.5:  # Set confidence threshold
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, model.names[int(class_id)], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                time_now = get_current_time()
                location = get_location()
                message = f"🚨 Weapon Detected! 🚨\n\nLocation: {location}\nTimestamp: {time_now}"
                bot.sendMessage(chat_id=TELEGRAM_CHAT_ID, text=message)
                # Optionally, you can send the frame as well
                cv2.imwrite("weapon_alert_frame.jpg", frame)
                bot.sendPhoto(chat_id=TELEGRAM_CHAT_ID, photo=open('weapon_alert_frame.jpg', 'rb'))
                break



    # Perform violence detection
    is_violence_detected = detect_violence(frame)

    # Process violence detection results and send alerts
    if is_violence_detected:
        # Draw "Violence" label on the frame
        cv2.putText(frame, "Violence", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)

        # Send Telegram alert for violence detected
        time_now = get_current_time()
        location = get_location()
        violence_message = f"🚨 Violence Detected! 🚨\n\nLocation: {location}\nTimestamp: {time_now}"
        bot.sendMessage(chat_id=TELEGRAM_CHAT_ID, text=violence_message)
        # Optionally, you can send the frame as well
        cv2.imwrite("violence_alert_frame.jpg", frame)
        bot.sendPhoto(chat_id=TELEGRAM_CHAT_ID, photo=open('violence_alert_frame.jpg', 'rb'))

    # Display the annotated video
    cv2.imshow('Real-time Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
