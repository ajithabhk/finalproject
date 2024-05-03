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
TELEGRAM_BOT_TOKEN = ''
TELEGRAM_CHAT_ID = ''

# Initialize Telegram Bot
bot = telepot.Bot(token=TELEGRAM_BOT_TOKEN)

# Load YOLO model for handgun detection
handgun_model = YOLO('runs/detect/train5/weights/best.pt')




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

    # Return True if violence is detected, False otherwise
    return preds[0][0] > 0.5


# Capture video from laptop camera
cap = cv2.VideoCapture(0)  # 0 indicates the default camera of your system

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform handgun detection
    handgun_results = handgun_model(frame)

    # Check if handgun detection results are not empty
    if handgun_results:
        # Process handgun detection results and send alerts
        for det in handgun_results:
            if len(det) >= 6:  # Check if det contains at least 6 elements
                x1, y1, x2, y2, conf, class_id = det[:6]  # Extract first 6 elements
                if conf > 0.5:  # Set confidence threshold
                    # Send Telegram alert for weapon detected
                    time_now = get_current_time()
                    location = get_location()
                    weapon_message = f"🚨 Weapon Detected! 🚨\n\nLocation: {location}\nTimestamp: {time_now}"
                    bot.sendMessage(chat_id=TELEGRAM_CHAT_ID, text=weapon_message)
                    # Optionally, you can send the frame as well
                    cv2.imwrite("weapon_alert_frame.jpg", frame)
                    bot.sendPhoto(chat_id=TELEGRAM_CHAT_ID, photo=open('weapon_alert_frame.jpg', 'rb'))
                    break  # Break out of loop after sending one alert for each detected weapon



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
