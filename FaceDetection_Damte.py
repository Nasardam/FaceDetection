import cv2
import os

#Import haar cascade xml file
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Open Front Camera
Detector = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame and convert the image to grayscale for easier computation
    ret, frames = Detector.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
# Draw rectangles on the detected objects
    for (x, y, width, height) in faces:
        cv2.rectangle(frames, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 2)

    # Display the results
    cv2.imshow('Video', frames)

    # Waiting for a key for image to close, adding the break statement to end the face detection screen
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
Detector.release()
cv2.destroyAllWindows()