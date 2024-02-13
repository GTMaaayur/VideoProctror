from flask import Flask, Response, render_template
import cv2
import time
import google.ai.generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
import os
import threading
from queue import Queue
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

load_dotenv()

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
login_manager = LoginManager()
login_manager.init_app(app)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

frame_queue = Queue()

Base = declarative_base()

class User(UserMixin, Base):
   __tablename__ = 'users'

   id = Column(Integer, primary_key=True)
   username = Column(String, unique=True)
   password = Column(String)

class ViolationData(Base):
   __tablename__ = 'violation_data'

   timestamp = Column(DateTime, primary_key=True, default=datetime.now)
   user_id = Column(Integer, ForeignKey('users.id'))
   earbuds = Column(Integer, default=0)
   mobile_phone = Column(Integer, default=0)
   camera = Column(Integer, default=0)
   gaze = Column(Integer, default=0)
   headpose = Column(Integer, default=0)
   talking_multiple_person = Column(Integer, default=0)

# SQLAlchemy configuration
db_url = 'sqlite:///violation_data.db'
engine = create_engine(db_url, echo=True)
Base.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()

def save_data(user_id, timestamp, counts):
   # Save violation counts and timestamp to the database
   violation_data = ViolationData(
       user_id=user_id,
       timestamp=timestamp,
       earbuds=counts['earbuds'],
       mobile_phone=counts['mobile_phone'],
       camera=counts['camera'],
       gaze=counts['gaze'],
       headpose=counts['headpose'],
       talking_multiple_person=counts['talking_multiple_person']
   )
   session.add(violation_data)
   session.commit()

def generate_frames(user_id):
   cap = cv2.VideoCapture(0)
   frame_delay = 0.1
   if not cap.isOpened():
       raise Exception("Error opening Camera")
   try:
       while True:
           ret, frame = cap.read()
           if not ret:
               raise Exception("Error reading the frame from the camera")
           image_byte = cv2.imencode('.jpg', frame)[1].tobytes()

           timestamp = datetime.now()

           response = model.generate_content(
               glm.Content(
                   parts=[
                       glm.Part(text=f"""Context: You are the video proctor for live video feed from the webcam of user {user_id}. Task:
                                 Check for any violations of the following rules:

                                       1. No earbuds should be present in the frame.
                                       2. No mobile phones should be present in the frame.
                                       3. No cameras should be present in the frame.
                                       4. The candidate's gaze should remain within the screen at all times.
                                       5. The candidate's head pose should be within the screen at all times.
                                       6. Should not talk to someone, and no multiple persons detected.

                                       Count the number of times each violation occurs. Answer:
                                       1) Earbuds violation: {violation_counts['earbuds']}
                                       2) Mobile phone violation: {violation_counts['mobile_phone']}
                                       3) Camera violation: {violation_counts['camera']}
                                       4) Gaze violation: {violation_counts['gaze']}
                                       5) Headpose violation: {violation_counts['headpose']}
                                       6) Talking violation or multiple person
                                       """),
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

           # Update violation counts based on the response
           violation_counts['earbuds'] += part.text.count('earbuds violation')
           violation_counts['mobile_phone'] += part.text.count('mobile phone violation')
           violation_counts['camera'] += part.text.count('camera violation')
           violation_counts['gaze'] += part.text.count('gaze violation')
           violation_counts['headpose'] += part.text.count('headpose violation')
           violation_counts['talking_multiple_person'] += part.text.count('talking violation') + part.text.count('multiple person violation')

           # Save counts along with timestamp to the database
           save_data(user_id, timestamp, violation_counts)

           time.sleep(frame_delay)

           _, buffer = cv2.imencode('.jpg', frame)
           frame = buffer.tobytes()
           frame_queue.put(frame)
   except Exception as e:
       print("Error:", str(e))
   finally:
       cap.release()

def generate_frames_thread(user_id):
   threading.Thread(target=generate_frames, args=(user_id,), daemon=True).start()

@app.route('/login', methods=['GET', 'POST'])
def login():
   if current_user.is_authenticated:
       return redirect(url_for('home'))
   form = LoginForm()
   if form.validate_on_submit():
       user = User.query.filter_by(username=form.username.data).first()
       if user and user.check_password(form.password.data):
           login_user(user, remember=form.remember.data)
           return redirect(url_for('home'))
       else:
           flash('Incorrect username or password.')
   return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
   logout_user()
   return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
   user_id = current_user.id
   generate_frames_thread(user_id)
   return render_template('home.html')

@app.route('/proctor', methods=['GET'])
@login_required
def proctor():
   return Response(generate_frames_generator(), content_type='multipart/x-mixed-replace; boundary=frame')

def generate_frames_generator():
   while True:
       frame = frame_queue.get()
       yield (b'--frame\r\n'
              b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000, threaded=True)