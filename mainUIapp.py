import streamlit as st
import sqlite3
from sqlite3 import Error
import cv2
from PIL import Image
from ultralytics import YOLO
import telepot
from datetime import datetime
import pytz
import numpy as np
from keras.models import load_model

# Load violence detection model
violence_model = load_model('modelnew.h5')

# Load YOLO model for handgun detection
model = YOLO('runs/detect/train5/weights/best.pt')

# Telegram Bot token and chat ID
TELEGRAM_BOT_TOKEN = '6549703008:AAGAC1nKsf4jrj4w-opTZeoc66JaHB-nGiY'
TELEGRAM_CHAT_ID = '-1002002290095'

# Initialize Telegram Bot
bot = telepot.Bot(token=TELEGRAM_BOT_TOKEN)

# Function to create a database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

# Function to create a new user
def create_user(conn, username, password):
    sql = ''' INSERT INTO users(username,password)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (username, password))
    conn.commit()
    return cur.lastrowid

# Function to check if a username exists in the database
def check_username_exists(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    rows = cur.fetchall()
    return len(rows) > 0

# Function to authenticate a user
def authenticate_user(conn, username, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    rows = cur.fetchall()
    return len(rows) > 0

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
    return preds[0][0] > 0.6

# Function to display video stream and perform detection
def display_video_stream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform handgun detection
        results = model(frame)
        boxes = results[0].boxes
        if boxes is not None:
            for det in boxes:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().tolist()
                conf = det.conf[0].item()
                class_id = det.cls[0].item()
                if conf > 0.5:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, model.names[int(class_id)], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    time_now = get_current_time()
                    location = get_location()
                    message = f"🚨 Weapon Detected! 🚨\n\nLocation: {location}\nTimestamp: {time_now}"
                    bot.sendMessage(chat_id=TELEGRAM_CHAT_ID, text=message)
                    cv2.imwrite("weapon_alert_frame.jpg", frame)
                    bot.sendPhoto(chat_id=TELEGRAM_CHAT_ID, photo=open('weapon_alert_frame.jpg', 'rb'))
                    break

        # Perform violence detection
        is_violence_detected = detect_violence(frame)

        if is_violence_detected:
            cv2.putText(frame, "Violence", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
            time_now = get_current_time()
            location = get_location()
            violence_message = f"🚨 Violence Detected! 🚨\n\nLocation: {location}\nTimestamp: {time_now}"
            bot.sendMessage(chat_id=TELEGRAM_CHAT_ID, text=violence_message)
            cv2.imwrite("violence_alert_frame.jpg", frame)
            bot.sendPhoto(chat_id=TELEGRAM_CHAT_ID, photo=open('violence_alert_frame.jpg', 'rb'))


        # Convert the frame to PIL Image
        pil_image = Image.fromarray(frame)

        # Display the annotated video
        st.image(pil_image, channels="BGR", use_column_width=True)

    cap.release()

# Main function
def main():
    st.title("Real-time Violence and Handgun Detection")

    # Connect to SQLite database
    conn = create_connection("users.db")

    # Create users table if it doesn't exist
    if conn is not None:
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
        """
        with conn:
            conn.execute(create_users_table)

    # Sidebar login/register section
    st.sidebar.title("Welcome to Real-time Detection App")
    login_section = st.sidebar.checkbox("Login")
    if login_section:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(conn, username, password):
                st.success("Login successful!")
                # Run real-time detection function
                display_video_stream()
            else:
                st.error("Invalid username or password.")

    register_section = st.sidebar.checkbox("Register Now")
    if register_section:
        st.subheader("Register")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            if not check_username_exists(conn, new_username):
                create_user(conn, new_username, new_password)
                st.success("Registration successful! You can now login.")
            else:
                st.error("Username already exists. Please choose a different one.")

# Run the app
if __name__ == "__main__":
    main()
