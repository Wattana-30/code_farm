import cv2

def getImg():
    video_capture = cv2.VideoCapture('http://localhost:1234/video')
    if not video_capture.isOpened():
        return False
    count = 0
    while True:
        count += 1
        _, frame = video_capture.read()
        img = frame
        if count > 10:
            break

    cv2.imshow("result", img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    video_capture.release()
    return img


print(getImg())