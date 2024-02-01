from flask import Flask, Response
import cv2
import time
import google.ai.generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
import os
import threading
from queue import Queue

load_dotenv()

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

frame_queue = Queue()

def generate_frames():
    cap = cv2.VideoCapture(0)#capture feed
    frame_delay = 1#will waited for 1 sec 
    if not cap.isOpened():
        raise Exception("Error opening Camera")
    try:
        while True:
            ret, frame = cap.read()#frame read
            if not ret:
                raise Exception("Error reading the frame from camera")
            image_byte = cv2.imencode('.jpg', frame)[1].tobytes()#.jpg firstmost fra
            response = model.generate_content(
                glm.Content(
                    parts=[
                        glm.Part(text="Context:"),
                        glm.Part(
                            inline_data=glm.Blob(
                                mime_type='image/jpeg',
                                data=image_byte
                            )
                        ),
                    ],
                ),
                stream=True
            )
            response.resolve()

            for part in response.parts:
                print(part.text)

            time.sleep(frame_delay)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame_queue.put(frame)
    except Exception as e:
        print("Error:", str(e))
    finally:
        cap.release()

def generate_frames_thread():
    threading.Thread(target=generate_frames, daemon=True).start()

@app.route('/proctor', methods=['GET'])
def proctor():
    return Response(generate_frames_generator(), content_type='multipart/x-mixed-replace; boundary=frame')

def generate_frames_generator():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    generate_frames_thread()
    app.run(host="0.0.0.0", port=5000, threaded=True)
