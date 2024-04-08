import pandas as pd
import torch
import cv2 as cv 
import os 
from collections import defaultdict

# Distance constants 
KNOWN_DISTANCE = 200 #CM

object_width = {"person":60, "chair":40, "couch":200}
indoor_object_set = {'couch','person','chair'}

# Model
# model_path = 'torch/hub/ultralytics_yolov5_master/'
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
# model = torch.hub.load(os.getcwd(), 'yolov5x')  # or yolov5m, yolov5l, yolov5x, etc.
# model = torch.load('torch/hub/ultralytics_yolov5_master', 'yolov5x')   # custom trained model

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
    return minResult 


def run(filename):
    directory_path = './TestImages'
    
    test_chair_1 = cv.imread(directory_path + '/' + filename)
    return distance_estimation(test_chair_1)

def formatRef(ref):
    object = ref.split(".")[0] #chair1_test
    object = object.split("_")[0] #chair1
    object = "".join([c for c in object if c.isalpha()]) #chair
    print("ref:", ref, "object:", object)
    return object 

if __name__ == "__main__":
    data = defaultdict(list)
    directory_path = './TestImages'
    for filename in os.listdir(directory_path):
        print(filename)
        data['filename'].append(filename)
        detected_object, distance = run(filename)
        data['detected_closest_object'].append(detected_object)
        data['real_object'].append(formatRef(filename))
        print("----------------------")
    ref_df = pd.DataFrame(data)
    ref_df.to_csv('test_data.csv', index=False)
    correct = ref_df[ref_df["detected_closest_object"] == ref_df["detected_closest_object"]].shape[0]
    print("accuracy: {}%".format(round(correct/ref_df.shape[0]*100), 2))

