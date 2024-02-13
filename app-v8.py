import cv2
import time
import google.ai.generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
import os
import threading
from queue import Queue
import logging
from flask import Flask, Response

load_dotenv()

app = Flask(__name__)
logging.basicConfig(filename='proctoring.log', level=logging.ERROR)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

frame_queue = Queue()
frame_accumulator = []

# Variables for sudden darkness detection
DARKNESS_THRESHOLD = 100  # Adjust as needed
DARKNESS_COUNTER_THRESHOLD = 5
darkness_counter = 0

def generate_frames():
    cap = cv2.VideoCapture(0)  # capture feed
    frame_delay = 0.2  # will wait for 0.2 sec
    accumulation_interval = 6  # accumulate frames for 6 seconds
    start_time = time.time()

    if not cap.isOpened():
        logging.error(msg='Error opening cap', exc_info=True)
        raise Exception("Error opening Camera")
    try:
        while True:
            ret, frame = cap.read()  # frame read
            if not ret:
                raise Exception("Error reading the frame from camera")
            
            frame_accumulator.append(frame)

            elapsed_time = time.time() - start_time
            if elapsed_time >= accumulation_interval:
                # Send batch requests
                batch_request(frame_accumulator)
                frame_accumulator.clear()
                start_time = time.time()

            time.sleep(frame_delay)

            # Check for sudden darkness
            if detect_sudden_darkness(frame):
                logging.error(msg='Sudden darkness detected')

    except Exception as e:
        logging.error(msg='Error reading frame', exc_info=True)
        print("Error:", str(e))
    finally:
        cap.release()

def batch_request(frames):
    try:
        batch_size = 10  
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            process_batch(batch_frames)
    except Exception as e:
        logging.error(msg=f'Error with batch frame:{str(e)}', exc_info=True)
    

def process_batch(frames):
    try:
        images_bytes = []
        for frame in frames:
            image_byte = cv2.imencode('.jpg', frame)[1].tobytes()
            images_bytes.append(image_byte)

        batch_request_parts = [
            glm.Part(text=f"""Context: You are the video proctor for live video feed from the webcam. Task:
                        check for any violations of the following rules:

                                1. No earbuds should be present in the frame.
                                2. No mobile phones should be present in the frame.
                                3. No cameras should be present in the frame.
                                4. The candidate's gaze should remain within the screen at all times.
                                5. The candidate's head pose should be within the screen at all times.
                                6. Should not talk to someone and no multiple person detected

                                Count the number of times each violation occurs. Answer:
                                1) earbuds violation:
                                2) mobile phone violation:
                                3) Camera violation:
                                4) Gaze violation:
                                5) Headpose violation:
                                6) talking violation or multiple person violation:
                                """)
        ]

        for image_byte in images_bytes:
            batch_request_parts.append(
                glm.Part(
                    inline_data=glm.Blob(
                        mime_type='image/jpeg',
                        data=image_byte
                    )
                )
            )

        response = model.generate_content(glm.Content(parts=batch_request_parts), stream=True)
        response.resolve()

        for part in response.parts:
            print(part.text)
    except Exception as e:
        logging.error(msg=f'Error with AI engine:{str(e)}', exc_info=True)
        
def detect_sudden_darkness(frame):
    global darkness_counter

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = cv2.mean(gray_frame)[0]

    if avg_brightness < DARKNESS_THRESHOLD:
        darkness_counter += 1
        if darkness_counter >= DARKNESS_COUNTER_THRESHOLD:
            darkness_counter = 0
            return True
    else:
        darkness_counter = 0

    return False
        
def generate_frames_thread():
    try:
        threading.Thread(target=generate_frames, daemon=True).start()
    except Exception as e:
        logging.error(msg=f'Error generating frame:{str(e)}', exc_info=True)

@app.route('/proctor', methods=['GET'])
def proctor():
    try:
        return Response(generate_frames_generator(), content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(msg=f'Error in proctor route:{str(e)}', exc_info=True)

def generate_frames_generator():
    try:
        while True:
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logging.error(msg=f'Error in generate_frame_geneartor:{str(e)}', exc_info=True)

if __name__ == '__main__':
    generate_frames_thread()
    app.run(host="0.0.0.0", port=5000, threaded=True)
