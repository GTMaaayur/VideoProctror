import google.ai.generativelanguage as glm
import google.generativeai as genai
import cv2
import time

# Configure the Generative AI service
genai.configure(api_key="AIzaSyCsd4miOUmXsn7XQqDfJq2A6FV1HxPbyzY")

# Initialize the Gemini-Pro-Vision model
model = genai.GenerativeModel('gemini-pro-vision')

# Open the video capture device
cap = cv2.VideoCapture(0)

# Set a delay between frames (adjust as needed)
frame_delay = 1  # in seconds

# Loop for real-time video analysis
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Break the loop if there's no more frames
    if not ret:
        break

    # Encode the frame to image bytes
    image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

    # Generate content using the model
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text="Context: You are an online exam proctor. Instructions: Check if the candidate is gazing out of the screen, wearing earbuds, having a mobile phone, or has the head posed away from the screen. Count such incidents and provide a real-time response. Also, report any violations in this frame."),
                glm.Part(
                    inline_data=glm.Blob(
                        mime_type='image/jpeg',
                        data=image_bytes
                    )
                ),
            ],
        ),
        stream=True
    )

    # Resolve the response
    response.resolve()

    # Display the original video feed (optional)
    cv2.imshow('Original Video Feed', frame)

    # Print real-time details
    for part in response.parts:
        print(part.text)

    # Add a delay between frames
    time.sleep(frame_delay)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device
cap.release()
cv2.destroyAllWindows()
