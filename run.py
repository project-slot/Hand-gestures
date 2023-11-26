import cv2
import pyautogui
from ultralytics import YOLO


# define a video capture object
vid = cv2.VideoCapture(0)
model = YOLO("./runs/detect/train/weights/best.pt")

ret = True
while ret:
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)

    result = model.predict(frame)
    if (result[0].boxes.xywh.numel()):
        box = result[0].boxes.xywh[0]
        x, y, w, h = (
            box[0],
            box[1],
            box[2],
            box[3],
        )  # x, y are the center coordinates.
        pyautogui.moveTo(x, y)
    frame = result[0].plot()

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
