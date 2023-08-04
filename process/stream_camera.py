from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import uvicorn

app = FastAPI()

camera = cv2.VideoCapture(0)
def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


if __name__ == '__main__':
    uvicorn.run('stream_camera:app', host='0.0.0.0', port=1234)
