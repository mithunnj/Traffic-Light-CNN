'''
Description: This contains all the code necessary to train the CNN model and save the model after training is done.
    This script will also handle all the post-processing required to prepare any new data added to the dataset (ex. holdout test data)
'''

import helper.update_dataset as update_dataset
import helper.yolo as yolo
import helper.crop as crop
from helper.split_data import split_dataset
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #for gradient descent
import json
torch.manual_seed(1) # set the random seed
import json
import cv2

# Global variables
CURRENT_DIR = os.getcwd()
CROP_IMGS = CURRENT_DIR + "/cropped_imgs"
YOLO_RESULT_FP = CURRENT_DIR + "/helper/yolo_result.json"
METADATA = list()

# Load metadata for the Bosch Trafficlight dataset as described in yolo_result.json
with open(YOLO_RESULT_FP, 'r') as read_file: 
  METADATA = json.load(read_file)
read_file.close()

# CNN archtecture for Traffic Light Classifier
class TrafficLightClassifier(nn.Module):
    def __init__(self):
        super(TrafficLightClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 2) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride 
        self.conv2 = nn.Conv2d(5, 10, 2) #in_channels, out_chanels, kernel_size
        self.fc1 = nn.Linear(360, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):        
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 360)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
    
        return x

def new_images():
    '''
    Given a set of new images, run through the initialization process to port the new data and prepare it for the rest of the pipeline.
    '''
    update_dataset.update_yaml_file() # Given a set of new images to add to the dataset, update the .yaml file
    update_dataset.update_img_size() # Take all the updates images and convert them to 720p - requirement for the rest of the CNN pipeline

    yolo.update_yolo() # Take the updated .yaml file and run YOLOv3 object detection on all the image dataset. Save the detected objects to yolo_result.json
    crop.crop_save() # Take the updated yolo_result.json file and crop all the detected object candidates. Save them in their respective directories

    return

def nn_parseable_data(relative_fp, metadata):
    '''
    param: relative_fp <str> - A single element from the split_data() resultant array of the form: /217352/217352_1.jpg. This is the relative path of the split data. The 
    underscore index for the image name refers to the objects that were detected in the image with the ID given in the directory name (ex. the original image was 217352.png in
    the dataset, and 217352_1.jpg refers to the first object that was detected in the image).
    param: metadata <dict> - A given entry in yolo_result.json
    return: <str>, <arr> - The absolute filepath of the cropped image, and the onehot encoding 
    '''
    absolute_fp = CROP_IMGS + relative_fp 
    string_split = relative_fp.split('/')

    string_split = [x for x in string_split if x] # Remove empty strings from array
    basename = string_split[0]
    object_num = int(string_split[-1].split('_')[-1].split('.')[0])

    for image in metadata:
        if basename == os.path.basename(image['filepath']).split(".")[0]:
            encoding = image['onehot_encod'][object_num]

            return absolute_fp, encoding

    return None, None

def train(model, data, metadata,num_epochs=2):
    criterion = nn.BCEWithLogitsLoss() # For binary classification
    optimizer = optim.SGD(model.parameters(), lr=0.0007, momentum=0.7)
    iters, loss_per_epoch, losses, train_acc, val_acc = [], [], [], [], []
    i = 0
    # training
    start_time = time.time() #start training timer
    print("training started...")                 
    for epoch in range(num_epochs):
      print("number of epocs total:", range(num_epochs))
      i += 1
      print("Epoch: ", i)
      print("test work 1")
      for image in data: # This is looping through each image of the data (in this case split_data['train']) that comes in the form: ['/225240/225240_0.jpg', '/225240/225240_3.jpg', '/225240/225240_1.jpg'....]
        try: 
            fp, onehot = nn_parseable_data(image, metadata) 
        except: 
            continue

        onehot_tensor = torch.Tensor([int(onehot)]) # Ensure the output of the onehot variable is of type integer, and needs to be a tensor to be fed into criterion        
        img = cv2.imread(fp) # Now that the image is loaded, now you can resize it
        img = resize_cv(img) # Overwrite the image variable with the resized version        
        
        out = model(img)              # forward pass
        loss = criterion(out, onehot_tensor) # Comparing model output to onehot *label* to produce loss value
        print("Out pred: {}, loss: {}, onehot: {}".format(out,loss, onehot_tensor)) #DEBUG
        loss.backward()               # backward pass (compute parameter updates)    
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch

      losses.append(loss)           #append loss to losses list 
      iters.append(epoch)    
      print('Train Accuracy: ', train_acc, 'Validation Accuracy: ', val_acc)
    end_time = time.time()   #end training timer
    print('training finished!!!')
    #Plotting Training curves
    print("Total training time: ", end_time - start_time)
    plt.title("Train loss")
    plt.plot(iters, losses, label="Train losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()  


UPDATE_DATA = False # Set this to true if you uploaded new test data to the dataset. This will ensure that the data is prepared for the rest of the piepline.
if UPDATE_DATA:
    new_images()

'''
NOTE: Uncomment codeblock below if you want to retrain CNN
model = TrafficLightClassifier()
split_data = split_dataset(CROP_IMGS)
train(model, split_data['train'], METADATA,num_epochs=20)
'''

    