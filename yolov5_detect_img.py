import torch
import cv2 as cv
import numpy as np
import pandas as pd
import os 

# Distance constants 
KNOWN_DISTANCE = 200 #CM

object_width = {"person":60, "chair":40, "couch":200}
indoor_object_set = {'couch','person','chair'}

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, etc.
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/best.pt')  # custom trained model

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


#detects objects and return relevant data (name, confidence score, boxwidth) in a dictionary
def object_detector(img):
    df = model(img).pandas().xyxy[0]
    data_list = [] 
    
    for ind in df.index:
        data_dict = dict()
        name = df['name'][ind]
        confidence = df['confidence'][ind]
        xmin = df['xmin'][ind]
        xmax = df['xmax'][ind]
        boxwidth = xmax - xmin
        data_dict["name"] = name 
        data_dict["boxwidth"] = boxwidth
        data_dict["confidence"] = confidence
        data_list.append(data_dict)
    return data_list  

#read from csv 
df = pd.read_csv("ref_data.csv")



def distance_estimation(img):
    data = object_detector(img) #[{'name': name, 'boxwidth': boxwidth, 'confidence': confidence}]
    print(data)
    minResult = ("", float('inf'))
    for item in data:
        if item['name'] in indoor_object_set:
            width_in_rf = float(df[df['name'] == item['name']].iloc[0]['boxwidth'])
            focal_length = focal_length_finder(KNOWN_DISTANCE, object_width[item['name']], width_in_rf)
            real_object_width = object_width[item['name']]
            width_in_frame = item['boxwidth']

            distance = distance_finder(focal_length, real_object_width, width_in_frame)
            minObject, minDistance = minResult 
            minResult = (minObject, minDistance) if minDistance <= distance else (item['name'], distance)
            
            print(f"Object: {item['name']} Boxwidth: {item['boxwidth']} Distance: {distance} cm")
    print(minResult)
def run(filename):
    directory_path = './TestImages'
    
    test_chair_1 = cv.imread(directory_path + '/' + filename)
    distance_estimation(test_chair_1)

if __name__ == "__main__":
    directory_path = './TestImages'
    for filename in os.listdir(directory_path):
        print(filename)
        run(filename)
        print("----------------------")

