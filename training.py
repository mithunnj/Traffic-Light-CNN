'''
Description: This contains all the code necessary to train the CNN model and save the model after training is done.
    This script will also handle all the post-processing required to prepare any new data added to the dataset (ex. holdout test data)
'''

import helper.update_dataset as update_dataset
import helper.yolo as yolo
import helper.crop as crop

def new_images():
    '''
    Given a set of new images, run through the initialization process to port the new data and prepare it for the rest of the pipeline.
    '''
    update_dataset.update_yaml_file() # Given a set of new images to add to the dataset, update the .yaml file
    update_dataset.update_img_size() # Take all the updates images and convert them to 720p - requirement for the rest of the CNN pipeline

    yolo.update_yolo() # Take the updated .yaml file and run YOLOv3 object detection on all the image dataset. Save the detected objects to yolo_result.json
    crop.crop_save() # Take the updated yolo_result.json file and crop all the detected object candidates. Save them in their respective directories

    return



UPDATE_DATA = False # Set this to true if you uploaded new test data to the dataset. This will ensure that the data is prepared for the rest of the piepline.
if UPDATE_DATA:
    new_images()



    