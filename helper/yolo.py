'''
Description: This helper function is meant to handle all the YOLOv3 object detection functionality.
    This script should be run after the .yaml file has been updated with all the new images in the dataset using the update_dataset.py helper funcion.
    Once the .yaml file is updated, this script will pull the new changes using parse_yaml(), run the the YOLOv3 network and then save all the detected objects per image
    in the yolo_result.json config file for further use.
'''

from helper.general_helper import parse_yaml
import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt
import json
import sys

BASE_DIR = os.path.dirname(os.getcwd())
YOLO_CONFIG = BASE_DIR + "/yolo-coco/yolov3.cfg"
YOLO_WEIGHTS = BASE_DIR + "/yolo-coco/yolov3.weights"
COCO_CLASSES = BASE_DIR + "/yolo-coco/coco.names"

CONFIDENCE = 0.5
THRESHOLD = 0.3
DET_OBJ = list()

IMAGE_COUNT, OBJ_COUNT = 0, 0 #DEBUG

def YOLO_processor(img_fp, net, labels, show_img=False):
    '''
    param: img_fp <str> - Filepath of the input image.
    param: net - the YOLO network that was loaded
    param: labels <arr> - List of all the possible classes from the YOLOv3 network
    param: show_img <bool> - If the user wants to see a visualization of the bounding box ontop of the image, they can set
        this parameter to True.
    return: None

    Given an input image, the YOLOv3 object detector will identify valid objects and create bounding boxes for the objects. We store  
    the detected objects and the coordinates of the bounding boxes in a datastructure (DET_OBJ) and use that for further processing later.
    Additionally, there is a onehot encoding version of the detected objects such that traffic lights are set to 1 and all other detected objects   
    are set to 0.
    '''
    global IMAGE_COUNT #DEBUG
    global OBJ_COUNT #DEBUG

    IMAGE_COUNT += 1 #DEBUG

    img_info = dict()
    img_info['filepath'] = img_fp
    img_info['items'] = list()
    img_info['onehot_encod'] = list()

    img = cv2.imread(img_fp)
    (H, W) = img.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:] #NOTE: I thought there was a typo here, original code was: detection[5:]
            #scores = detection
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID) 

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            item_metadata = dict()
            onehot_metadata = list()
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = (0, 0, 255) # Blue
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

            item_metadata['label'] = text # Detected item label and confidence
            item_metadata['bound_x'] = x # Bounding box x-coordinate
            item_metadata['bound_y'] = y # Bounding box y-coordinate
            item_metadata['bound_w'] = w # Bounding box width
            item_metadata['bound_h'] = h # Bounding box height

            OBJ_COUNT += 1 #DEBUG

            img_info['items'].append(item_metadata)

            if "traffic light" in text:
                img_info['onehot_encod'].append(1)
            else:
                img_info['onehot_encod'].append(0)

    if show_img:
        # show the output image
        cv2.imshow("Image", img)

    DET_OBJ.append(img_info)

    return

def backup_YOLO_result():
    '''
    Backup the datastructure information of all the detected objects and the bounding boxes for the Traffic Light dataset.
    This is to ensure that we don't have to spend time processing the information on every run and we can load the data instead.
    '''
    with open('./yolo_result.json', 'w') as write_file:
        json.dump(DET_OBJ, write_file)
    write_file.close()

    return


def update_yolo():
    dataset = parse_yaml()
    coco_classes = open(COCO_CLASSES).read().splitlines()
    net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)

    for data in dataset:
        YOLO_processor(data['fp'], net, coco_classes)

    #Backup contents of YOLO object detection processing for the images in the dataset
    #NOTE: Uncomment line below if you want to re-write the yolo_result.json backup file
    backup_YOLO_result()

    print('# of pictures processed: {}'.format(IMAGE_COUNT)) #DEBUG
    print('# of objects processed: {}'.format(OBJ_COUNT)) #DEBUG

    return
