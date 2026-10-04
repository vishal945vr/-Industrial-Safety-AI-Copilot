from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo26s.pt")

# Train the model on Construction-PPE dataset
model.train(data="construction-ppe.yaml", epochs=1, imgsz=640)
