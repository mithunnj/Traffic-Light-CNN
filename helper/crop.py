'''
Description: Given an updated yolo_result.json file, loop through the contents of the file and save all the candidate objects
    detected by the YOLOv3 object detection portion of the pipeline and save all the candidate images in their own directories.
'''

import cv2
import os
import json
import sys

DATASET = list()
CURRENT_DIR = os.path.dirname(os.getcwd())
CROP_IMGS = CURRENT_DIR + "/cropped_imgs"
YOLO_JSON = CURRENT_DIR + "/helper/yolo_result.json"

def crop_save():
    '''
    Take the result of the YOLOv3 step of the pipeline and crop the individual detected objects.
    '''

    # Load the YOLOv3 processing results
    with open(YOLO_JSON) as f: 
        DATASET = json.load(f)
    f.close()

    # Loop through each image in the processing results and crop the image and save it
    for data in DATASET:
        img = cv2.imread(data["filepath"])
        filename = os.path.basename(data["filepath"]).split('.')[0]
        custom_fp = CROP_IMGS + "/{}".format(filename) # Directory to contain this image's information
        file_name = custom_fp + "/file.name" # Store the original filepath of the image from the dataset
        count = 0 # Count to keep track of the cropped objects from a single image

        # Create the specific folder if it doesn't exist before moving forward in this pipeline
        if not os.path.isdir(custom_fp):
            os.mkdir(custom_fp)

        # For detected object in a single image, save the cropped version of the image with a specific filename
        for item in data['items']:
            x,y,w,h = item['bound_x'], item['bound_y'], item['bound_w'], item['bound_h']
            if x < 0 or y < 0 or w < 0 or h < 0:
                continue

            crop_img = img[y:y+h, x:x+w]    

            cv2.imwrite(custom_fp + "/{}_{}.jpg".format(filename, count), crop_img)
            count += 1

        # Store the original filepath of this image
        with open(file_name, "w+") as write_file:
            write_file.write(data["filepath"])
        write_file.close()

    return 
