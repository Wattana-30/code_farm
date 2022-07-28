import cv2
import datetime, numpy as np

def gstreamer_pipelineRGB(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_id=0 ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink wait-on-eos=True max-buffers=1 drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



# Video capturing from OpenCV gstreamer_pipelineRGB() , cv2.CAP_GSTREAMER
video_capture = cv2.VideoCapture(gstreamer_pipelineRGB(), cv2.CAP_GSTREAMER)
#gstreamer_pipelineRGB() , cv2.CAP_GSTREAMER
while True and video_capture.isOpened():
	return_key, frame = video_capture.read()
	if return_key:
		img = cv2.imwrite("picture01.jpg",frame)
		time1 = str(datetime.datetime.now())
		img = cv2.imread("picture01.jpg")
		img = cv2.resize(img, (400, 400))

		lower = np.array([0, 10, 10])
		upper = np.array([124, 255, 133])
		mask = cv2.inRange(img, lower, upper)
		result = cv2.bitwise_and(img, img, mask=mask)

		cv2.putText(img, time1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 0, 0), 1, cv2.LINE_AA)
		cv2.imshow("ori", img)
		cv2.imshow("split Color", mask)
		cv2.imshow("result", result)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
# Destroy all the windows
cv2.destroyAllWindows()

