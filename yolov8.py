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
def main():
    if(args[0] == "train"):
        results = model.train(
                                data='maimai.yaml', 
                                epochs=100, 
                                batch=1,
                                imgsz=640, 
                                save=True, 
                                save_period=30,
                                project="YOLOv8-ISDN6830-ObjectDetection"
    )
    elif(args[0] == "val"):
        metrics = model.val()  
        metrics.box.map    
        metrics.box.map50  
        metrics.box.map75  
        metrics.box.maps   

    elif(args[0] == "export"):
        model.export(format='onnx')

if __name__ == "__main__":
    main()