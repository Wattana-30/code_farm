import cv2


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


def getImg():
    count = 0
    video_capture = cv2.VideoCapture(
        gstreamer_pipelineRGB(), cv2.CAP_GSTREAMER)

    if not video_capture.isOpened():
        return False

    while True:
        count += 1
        _, frame = video_capture.read()

        if count > 100:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return frame
getImg()
