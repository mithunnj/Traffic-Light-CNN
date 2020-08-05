'''
Description:
'''

from training import TrafficLightClassifier
import os
import torch
from helper.split_data import split_dataset
from helper.general_helper import parse_yaml
from training import nn_parseable_data
import json
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Global variables
CURRENT_DIR = os.getcwd()
SAVED_MODEL = CURRENT_DIR + "/helper/savedmodel.pth"
CROP_IMGS = CURRENT_DIR + "/cropped_imgs"
DATASET = CURRENT_DIR + "/data/dataset_train_rgb"
YOLO_RESULT_FP = CURRENT_DIR + "/helper/yolo_result.json"
TRAIN_YAML_DIR = CURRENT_DIR + "/data/dataset_train_rgb/train.yaml"

# Print colours
CGREEN  = '\33[32m'
CEND    = '\33[0m'
CBLUE   = '\33[34m'
CRED    = '\033[91m'
CYELLOW = '\33[33m'

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

def bright_spot(rgb_image): 
    ''' Takes in a standardized RBG image and determines the location of the light, and will help determine what colour it is. ''' 

    def im_crop(image, lower_y, upper_y, lower_x, upper_x):
        '''Takes an image and specific crop dimensions and outputs a cropped image.'''
        crop_im = image[lower_y:upper_y, lower_x:upper_x]
        return crop_im

    def rgb_v(image):
        '''
        Takes an RGB image and converts it into an HSV image.
        '''
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 
        v_channel = hsv_image[:,:,2]
        return v_channel

    def red_green(rgb_image, position):
        '''
        Takes an image and double checks to determine if it is red or green.
        '''
        
        width = rgb_image.shape[0]
        height = rgb_image.shape[1]
        
        pixels = []
        
        for i in range(width):
            for j in range(height):
                pixels.append(rgb_image[i][j]) 
        
        total_red = 0
        total_green = 0

        for i in range(len(pixels)):
            total_red = pixels[i][0] + total_red

        for j in range(len(pixels)):
            total_green = pixels[i][1] + total_green
            
        area = width*height
        
        if position == 1:
            avg_red = total_red / area
            avg_green = total_green*0.445 / area
        else:
            avg_red = total_red*0.445 / area
            avg_green = total_green / area

        if avg_red > avg_green:
            state = 'red'
        else:
            state = 'green'
    
        return state

    def avg_brightness(v_channel):
        total_brightness = np.sum(v_channel)
        width = v_channel.shape[0]
        height = v_channel.shape[1]
        
        area = width*height
        avg_brightness = total_brightness / area

        return avg_brightness


    rgb_image = cv2.resize(rgb_image, (32, 32))  
    upper = im_crop(rgb_image, 0, 11, 0, 32)
    middle = im_crop(rgb_image, 12, 21, 0, 32)
    lower = im_crop(rgb_image, 22, 32, 0, 32)
    
    upper_v = rgb_v(upper)
    middle_v = rgb_v(middle)
    lower_v = rgb_v(lower)
    
    upper_brightness = avg_brightness(upper_v)
    middle_brightness = avg_brightness(middle_v)
    lower_brightness = avg_brightness(lower_v)

    values = []
    values.append(upper_brightness)
    values.append(middle_brightness)
    values.append(lower_brightness)
    
    max_value = max(values)
    counter = 1
    
    for i in values:
        if i == max_value:
            position = counter
        else:
            counter = counter + 1
    
    if position == 1:
        selected_im = upper
    elif position == 2:
        selected_im = middle
    else:
        selected_im = lower


    if position == 1:
        state = red_green(selected_im, position)
    elif position == 2:
        state = 'yellow'
    else:
        state = red_green(selected_im, position)


    return state

def get_accuracy(model, data_loader, metadata, original_data, demo=False, demo_fp=None):
    correct, total, colour_correct, colour_tot, pred_result = 0, 0, 0, 0, str()

    if not demo:

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

                # If we have a valid traffic light candidate
                if pred == 1:
                    rgb_image = cv2.imread(fp)
                    pred_result = bright_spot(rgb_image)
                    colour_tot += 1

                    basename = fp.split('/')[-2]

                    for data in original_data:
                        if basename == data['fp'].split('/')[-1].split('.')[0]:
                            if pred_result in [x.lower() for x in data['labels'].keys()]:
                                colour_correct += 1

        
        total = len(data_loader)
        
        print("Percentage correct classification by CNN: {} %".format((correct / total)*100))
        print("Percentage correct classification by colour detector: {} %".format((colour_correct / colour_tot)*100))
    else:
        split_data_opt = ['train', 'val', 'test']
        file_id = demo_fp.split('/')[-1].split('.')[0]
        image_info = None

        # Find the corresponding information
        for option in split_data_opt:
            for image in data_loader[option]:
                if file_id in image:
                    image_info = [x for x in image.split('/') if x][0]

        # Generate filepath
        filepath = os.getcwd() + "/cropped_imgs/{}".format(image_info)
        files = list(os.listdir(filepath))
        files.remove("file.name")

        for file_name in files:
            fp, onehot = nn_parseable_data("/{}/{}".format(image_info, file_name), metadata) #NOTE: Your onehot encoding error could have been because some of the values returned were strings instead of integers, so thats why I've converted the type directly here.

            onehot_tensor = torch.Tensor([int(onehot)]) # Ensure the output of the onehot variable is of type integer, and needs to be a tensor to be fed into criterion        
            img = cv2.imread(fp) # Now that the image is loaded, now you can resize it
            img = resize_cv(img) # Overwrite the image variable with the resized version   
        
            output = model(img)
            output = output.detach().numpy()[0]
            onehot_as_int = onehot_tensor.detach().numpy()[0]
            
            if output <= 0:
                pred = 0
            else:
                pred = 1
        
            if pred == onehot_as_int:
                correct += 1

                # If we have a valid traffic light candidate
                if pred == 1:
                    rgb_image = cv2.imread(fp)
                    pred_result = bright_spot(rgb_image)
                    colour_tot += 1

                    basename = fp.split('/')[-2]

                    for data in original_data:
                        if basename == data['fp'].split('/')[-1].split('.')[0]:
                            if pred_result in [x.lower() for x in data['labels'].keys()]:
                                print("Pred_result: ", pred_result)
                                colour_correct += 1

        total = len(files)
        print("Percentage correct classification by CNN: {} %".format((correct / total)*100))
        print("Percentage correct classification by colour detector: {} %".format((colour_correct / colour_tot)*100))

        command_str, colour = "", ""
        if "green" in pred_result:
            command_str = "GO THROUGH"
            color = CGREEN
        elif "yellow" in pred_result:
            command_str = "SLOW DOWN"
            color = CYELLOW
        else:
            command_str = "STOP"
            color = CRED
        
        print(color + "\nVEHICLE COMMAND: {}".format(command_str) + CEND)

    return 



# Load partitioned dataset
split_data = split_dataset(CROP_IMGS)

# Load original dataset information
yaml_info = parse_yaml(yaml_fp=TRAIN_YAML_DIR)

# Define model class and load previously trained CNN 
model = TrafficLightClassifier()
model = torch.load(SAVED_MODEL)

# Ask user for demo and prepare
user_demo = input("\nWould you like to do a DEMO [y/n]: ").strip()
demo_image = {
    "red"       : "/rgb/train/2015-10-05-16-02-30_bag/579386.png",
    "yellow"    : "/rgb/train/2015-10-05-16-02-30_bag/625182.png",
    "green"     : "/rgb/train/2015-10-05-16-02-30_bag/627710.png"
}
if user_demo == "y":
    user_demo = True
    user_colour = input("\nWhat colour would you like to select:\nRed\nGreen\nYellow\n:").strip().lower()
    basename = demo_image[user_colour].split("/")[-1].split(".")[0]

    print(CGREEN + "\nStarting demo for:\n-Colour: {}\n-Image: {}\n-Detected Objects: {}\n".format(user_colour, str(DATASET + demo_image[user_colour]), CROP_IMGS + "/{}".format(basename)) + CEND)
    get_accuracy(model, split_data, METADATA, yaml_info, demo=user_demo, demo_fp=demo_image[user_colour])
else:
    get_accuracy(model, split_data['test'], METADATA, yaml_info)
