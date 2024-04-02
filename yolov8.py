import comet_ml
from ultralytics import YOLO
import os
import sys

args = sys.argv[1:]

os.environ["COMET_API_KEY"] = "aj85EqZJE9ZTII3FMaReNU1kw"

#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('best.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

if(args[0] == "train"):
    results = model.train(
                            data='maimai.yaml', 
                            epochs=100, 
                            batch=8,
                            imgsz=640, 
                            save=True, 
                            save_period=30,
                            project="YOLOv8-ISDN6830-ObjectDetection"
    )
elif(args[0] == "val"):
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category


