import cv2

# Facial detection using haarcascade classifiers

# loading haarcascade classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam feed
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame at a time
    ret, frame = video_capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5)

    # Draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

    # Display frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


