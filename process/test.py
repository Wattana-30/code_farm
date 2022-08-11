import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break
    else:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        cv2.imshow('show',buffer)
        print(success)