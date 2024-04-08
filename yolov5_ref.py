import torch
import cv2 as cv
import numpy as np
import pandas as pd
import os 
from ultralytics import YOLO
from collections import defaultdict 

# Distance constants 
KNOWN_DISTANCE = 200 #CM

object_width = {"person":60, "chair":40, "couch":200}
indoor_object_set = {'couch','person','chair'}

#  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
# model_path = '~/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth'
# model = deeplabv3_resnet101(pretrained=True)
# model.load_state_dict(torch.load(model_path))
# model = YOLO("yolov5l.pt")
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5l, yolov5x, etc.
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/best.pt')  # custom trained model
ref_df = defaultdict(list)


#detects objects and return relevant data (name, confidence score, boxwidth) in a dictionary
def object_detector_obj(img, object):
    print(model(img))
    df = model(img).pandas().xyxy[0]
    for ind in df.index:
        name = df['name'][ind]
        if name == object:
            xmin = df['xmin'][ind]
            xmax = df['xmax'][ind]
            boxwidth = xmax - xmin
            ref_df["name"].append(name)
            ref_df["boxwidth"].append(boxwidth)
            return 
    return 

def filterObj(filename): #chair1.jpg
    object = filename.split(".")[0] #chair1
    object = "".join([c for c in object if c.isalpha()]) #chair
    return object 
        
def run(filename):
    directory_path = './RefImages'
    df = {}
    test_chair_1 = cv.imread(directory_path + '/' + filename)
    object_detector_obj(test_chair_1, filterObj(filename))

if __name__ == "__main__":

    directory_path = './RefImages'
    for filename in os.listdir(directory_path):
        print(filename)
        run(filename)
        print("----------------------")
    ref_df = pd.DataFrame(ref_df)
    ref_df.to_csv("ref_data.csv")

