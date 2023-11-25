from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(resume = True)  # data="data.yaml", epochs=2, classes=[3,19,14,17,12,13,9,10]
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export()  # export the model to ONNX format