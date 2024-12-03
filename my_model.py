from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')

results = model.train(data='config.yaml', epochs=30, imgsz=640)