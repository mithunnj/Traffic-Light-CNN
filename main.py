'''
Description:
'''

from training import TrafficLightClassifier
import os
import torch
from helper.split_data import split_dataset
from training import nn_parseable_data
import json
import sys
import cv2

# Global variables
CURRENT_DIR = os.getcwd()
SAVED_MODEL = CURRENT_DIR + "/helper/savedmodel.pth"
CROP_IMGS = CURRENT_DIR + "/cropped_imgs"
YOLO_RESULT_FP = CURRENT_DIR + "/helper/yolo_result.json"

# Load metadata for the Bosch Trafficlight dataset as described in yolo_result.json
METADATA = list()
with open(YOLO_RESULT_FP, 'r') as read_file: 
  METADATA = json.load(read_file)
read_file.close()

def resize_cv(image):
  """
  param: <opened image>- A image that has been opened is fed into resize_cv() where the image is then converted to an image of specified
    dimensions in this case 28, 8

  returns <opened image>- Returns *same* opened image with new dimensions which can then be saved over
  """
  image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_LINEAR)
  image = torch.from_numpy(image)
  reformatted_image = image.reshape([1, 3, 28, 28]) 
  return reformatted_image


def get_accuracy(model, data_loader, metadata):
    correct = 0
    total = 0
    skip = 0 # DEBUG
    for image in data_loader:
        fp, onehot = nn_parseable_data(image, metadata) #NOTE: Your onehot encoding error could have been because some of the values returned were strings instead of integers, so thats why I've converted the type directly here.

        onehot_tensor = torch.Tensor([int(onehot)]) # Ensure the output of the onehot variable is of type integer, and needs to be a tensor to be fed into criterion        
        img = cv2.imread(fp) # Now that the image is loaded, now you can resize it
        img = resize_cv(img) # Overwrite the image variable with the resized version   
        print('One hot tensor: ', onehot_tensor)
      
        output = model(img)
        print('Output here: ', output)
        output = output.detach().numpy()[0]
        onehot_as_int = onehot_tensor.detach().numpy()[0]
        
        if output <= 0:
         pred = 0
        else:
          pred = 1
    
        if pred == onehot_as_int:
          correct += 1
    
    total = len(data_loader)
    print('Correct: ', correct)
    print('Total: ', total)
    return print((correct / total)*100,'%')

# Load partitioned dataset
split_data = split_dataset(CROP_IMGS)

# Define model class and load previously trained CNN 
model = TrafficLightClassifier()
model = torch.load(SAVED_MODEL)

# Test model accuracy
get_accuracy(model, split_data['val'], METADATA)
