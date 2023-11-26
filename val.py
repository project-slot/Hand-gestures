from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")




metrics = model.val()  # no arguments needed, dataset and settings remembered