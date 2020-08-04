'''
Description: If you have new image data to add to the dataset, run the contents of this helper function to assist you with adding the info to the .yaml file that
    is accessed throughout this software pipeline. Additionally, normalize all the image data by converting the images to 720p (1280px by 720px)
'''

import yaml
import os
import sys
import cv2
import matplotlib.pyplot as plt

BASEDIR = os.getcwd().split('helper')[0] # Remove the helper directory from the project base filepath
YAML_FP = BASEDIR + "data/dataset_train_rgb/train.yaml"
UPDATE_BASE_FP = BASEDIR + "data/dataset_train_rgb/rgb/test"

def open_image(filepath):
    '''
    param: filepath <str> - Given an image filepath, open the image for the user.
    return: None
    '''
    filepath = UPDATE_BASE_FP + "/{}".format(os.path.basename(filepath))
    img = cv2.imread(filepath)
    plt.imshow(img)

    return


def update_yaml_file():
    '''
    Given a set of new images to add to the dataset, update the .yaml file to help with the rest of the pipeline.
    This will update the yaml file with empty bounding box information. These are for cases where custom non-annotated
    data is added to the dataset.
    '''
    yaml_contents = list()
    update_files_list = os.listdir(UPDATE_BASE_FP) # Get a list of all the new images to add to the yaml file.

    with open(YAML_FP, 'r') as read_file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        yaml_contents = yaml.load(read_file, Loader=yaml.FullLoader)
    read_file.close()

    with open(YAML_FP, 'w') as write_file:
        for update_file in update_files_list:
            path = "./rgb/test/{}".format(update_file) # Filepath for the new image in the test dataset

            # After the image is shown to the user, ask the user how many traffic lights are in the image
            open_image(path) # Open the image
            #num_trafficlight = input('Enter the number of traffic lights in {}: '.format(path))
            num_trafficlight = 1
            boxes_content = list()

            # Ask the user to provide information about the traffic lights in the image.
            # Since this is data for the holdout test set, set default values for the rest of the parameters
            for i in range(int(num_trafficlight)):
                box_metadata = dict()
                label_color = input('What is the colour of the traffic light (NOTE: Capitalize the first letter)\n{}\n: '.format(path))
                if label_color == "g":
                    box_metadata['label'] = "Green"
                elif label_color == "r":
                    box_metadata['label'] = "Red"
                else:
                    box_metadata['label'] = "Yellow"
                box_metadata['occluded'] = False
                box_metadata['x_max'] = 1
                box_metadata['x_min'] = 1
                box_metadata['y_max'] = 1
                box_metadata['y_min'] = 1

                # Store the information
                boxes_content.append(box_metadata)

            content = dict()
            content['boxes'] = boxes_content
            content['path'] = path
            yaml_contents.append(content)

        write_result = yaml.dump(yaml_contents, write_file)

    write_file.close()

    return

def update_img_size():
    '''
    Given a list of new files, resize it to 720p to adjust for the rest of the 
    data processing pipeline.
    '''
    update_files_list = os.listdir(UPDATE_BASE_FP)

    # Loop through every new image and resize to 720p
    for image in update_files_list:
        fp = "{}/{}".format(UPDATE_BASE_FP, image)
        load_img = cv2.imread(fp)
        resized_image = cv2.resize(load_img, (1280, 720)) 
        cv2.imwrite(fp, resized_image)

    return


#update_yaml_file() # Uncomment this line to add new images to the .yaml file for training/testing purposes.
#update_img_size() # Uncomment this line to be able to resize images to 720p.
