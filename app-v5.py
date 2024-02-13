from flask import Flask,jsonify,request,Response
import cv2
import time
import google.ai.generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

app=Flask(__name__)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model=genai.GenerativeModel('gemin-pro-vision')

@app.route('/proctor',methods=['GET'])
async def proctor():
  return Response(generate_frames(),content_type='multipart/x-mixed-replace; boundary=frame')
async def generate_frames():
  cap=cv2.VideoCapture(0)
  frame_delay=1
  if not cap.isOpened():
    raise Exception("Error opening Camera")
  try:
    while True:
      ret,frame=cap.read()
      if not ret:
        raise Exception("Error reading the frame from camera")
      image_byte=cv2.imencode('.jpg',frame)[1].tobytes()
      response=model.generate_content(
        glm.Content(
          parts=[
            glm.Part(text="Context:"),
            glm.Part(
              inline_date=glm.Blob(
                mime_type='image/jpeg',
                data=image_byte
              )
            ),
          ],
        ),
        stream=True
      )
      await response.resolve()
      for part in response.parts():
        print(part.text)
      time.sleep(frame_delay)

      _,buffer=cv2.imencode('.jpg',frame)
      frame=buffer.tobytes()
      yield(b'--frame\r\n'
         b'Content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')
  except Exception as e:
     print("Error:", str(e))
  finally:
    cap.release()
if __name__=='__main__':
  import uvicorn
  uvicorn.run(app=app,host="0.0.0.0",port=5000,workers=4)
