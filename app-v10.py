import cv2
import time
import datetime
import google.ai.generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
import os
import threading
import logging
from queue import Queue

from flask import Flask, Response, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

app = Flask(__name__)

logging.basicConfig(filename='proctoring.log', level=logging.ERROR)
logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')
modelt = genai.GenerativeModel('gemini-pro')

frame_queue = Queue()
frame_accumulator = []

engine = create_engine('sqlite:///proctoring.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class Response(Base):
    __tablename__ = 'responses'
    id = Column(Integer, primary_key=True)
    response_text = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.now)

    
Base.metadata.create_all(engine)

DARKNESS_THRESHOLD = 100
DARKNESS_COUNTER_THRESHOLD = 5
darkness_counter = 0

def generate_frames(duration):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_delay = 0.2
    start_time = time.time()

    if not cap.isOpened():
        logging.error(msg='Error opening cap', exc_info=True)
        raise Exception("Error opening Camera")
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error reading the frame from camera")
            
            frame_accumulator.append(frame)

            time.sleep(frame_delay)

            if detect_sudden_darkness(frame):
                logging.error(msg='Sudden darkness detected')

            
            process_frame(frame)

    except Exception as e:
        logging.error(msg='Error reading frame', exc_info=True)
        print("Error:", str(e))
    finally:
        cap.release()

def process_frame(frame):
    try:
        image_byte = cv2.imencode('.jpg', frame)[1].tobytes()

        request_parts = [
            glm.Part(text=f"""Context: You are the video proctor for live video feed from the webcam. Task:
                        check for any violations of the following rules:

                                1. No earbuds should be present in the frame.
                                2. No mobile phones should be present in the frame.
                                3. No cameras should be present in the frame.
                                4. The candidate's gaze should remain within the screen at all times.
                                5. The candidate's head pose should be within the screen at all times.
                                6. Should not talk to someone and no multiple person detected

                                Count the number of times each violation occurs. 
                                Answer BY percentage violation by each above types:
                                earbuds:
                                mobile phone:
                                camera:
                                gaze violation:
                                multiple persons:
                                talking to someone:
                                """)
        ]

        batch_request_parts = request_parts + [
            glm.Part(
                inline_data=glm.Blob(
                    mime_type='image/jpeg',
                    data=image_byte
                )
            )
        ]

        response = model.generate_content(glm.Content(parts=batch_request_parts), stream=True)
        response.resolve()

        for part in response.parts:
            response_text = part.text
            store_response(response_text)

    except Exception as e:
        logging.error(msg=f'Error processing frame: {str(e)}', exc_info=True)

def store_response(response_text):
    try:
        response = Response(response_text=response_text)
        session.add(response)
        session.commit()
    except Exception as e:
        logging.error(msg=f'Error saving reponse :{str(e)}')
def detect_sudden_darkness(frame):
    global darkness_counter
    try:
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
    except Exception as e:
        logging.error(msg=f"Error in darkness detection:{str(e)}")

def generate_frames_thread():
    duration = 60 # 1 minutes
    try:
        threading.Thread(target=generate_frames, args=(duration,), daemon=True).start()
    except Exception as e:
        logging.error(msg=f'Error generating frame:{str(e)}', exc_info=True)

@app.route('/proctor', methods=['GET'])
def proctor():
    try:
        return Response(generate_frames_generator(), content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(msg=f'Error in proctor route:{str(e)}', exc_info=True)

@app.route('/generate_report', methods=['GET'])
def generate_report():
    try:
        generate_malpractice_report()
        return jsonify({'message': 'Malpractice report generated successfully'}), 200
    except Exception as e:
        logging.error(msg=f'Error generating malpractice report: {str(e)}', exc_info=True)
        return jsonify({'error': f'Failed to generate malpractice report:{str(e)}'}), 500
    
def generate_frames_generator():
    try:
        while True:
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logging.error(msg=f'Error in generate_frame_geneartor:{str(e)}', exc_info=True)

def generate_malpractice_report():
    try:
        response_texts = session.query(Response.response_text).all()
        
        concatenated_text = '\n'.join(row[0] for row in response_texts)
        concatenated_text = f'""""""\n{concatenated_text}\n""""""'
        prompt = """You are a report generation assistant. Write a report  determining if the candidate has committed malpractice throughout the exam. Here are the logs:
        Guidelines:keep some Standard Tolarance value as 20% max if exceeded make report as malpractise found
                                RETURN IN THIS FORM ONLY(explicitly look for moblie phone, camera,earphones or earbuds  ):
                                        company name:
                                        Malpractise:found/not found (based on toleration value)
                                        Exam date:
                                        type of violation:name of type and  proof from logs
        """
        
        report_content = [prompt, concatenated_text]
        
        report_text = modelt.generate_content(report_content, stream=False)
        
        for part in report_text.parts:
            report_text = part.text
            save_generated_report(report_text=report_text,filename='report.txt')
            print(report_text)


    except Exception as e:
        logging.error(msg=f'Error generating malpractice report: {str(e)}', exc_info=True)
def save_generated_report(report_text,filename):
    try:
        with open(filename,mode="w") as f:
            f.write(report_text)
    except Exception as e:
        logging.error(msg=f'Error saving the file:{str(e)}',exc_info=True)

if __name__ == '__main__':
    generate_frames_thread()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
