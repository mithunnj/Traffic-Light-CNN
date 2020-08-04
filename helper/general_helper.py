'''
Description: These are general and widely used helper functions that are just all bundled together into a single file.
  These were part of the early development of the processing pipeline. 
'''

import os
import yaml
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import cv2
import numpy as np
import math

BASE_DIR = os.path.dirname(os.getcwd())
TRAIN_YAML_DIR = BASE_DIR + "/data/dataset_train_rgb/train.yaml"

def parse_yaml(yaml_fp=TRAIN_YAML_DIR, simple_off=False, simple_dir=False, simple_label=False):
    '''
    param: yaml_fp <str> - This is the absolute filepath of the .yaml file containing the anotated metadata.
        Default is set to the training set .yaml file.
    param: simple_off, simple_dir, simple_label <bool> - This will only return images from the dataset that contian a single label to simplify
        the development process. Otherwise, it will include examples of images with multiple classification options. Specifically, you can specify
        to include/remove all images with the label "off" or images that contain directional traffic light labels like "RedLeft," "GreenLeft", etc. 
    return: return_obj <arr> - Each element is a dict containing the filepath and the identified traffic lights for each image.
    Description: Parse the .yaml file and return a list of all the images that contain labelled information
        along with the labelled traffic light colour(s).
    '''
    yaml_contents = list()

    # Parse and store contents of .yaml annotations.
    with open(yaml_fp, 'r') as contents:
        yaml_contents = yaml.safe_load(contents) # Converts contents of .yaml file into a parseable list format, where each element is a json
    contents.close()

    # Keep images of the .yaml file that contain a valid label.
    return_obj = list() # Datastructure that will be returned by the function

    for metadata in yaml_contents:
        val_fp = str() # Placeholder to store the filepath of a valid image.
        img_contents = dict() # For a valid image, this will contain the fp and the image labels.

        # If there is missing metadata, and the image is not part of a user added hold out test set from the 'test/' directory, then skip
        if (not metadata['boxes'] or not metadata['path']) and not 'test/' in metadata['path']:
          continue
        else:
            val_fp = BASE_DIR + '/data/dataset_train_rgb/' + os.path.relpath(metadata.get('path'), './') # Remove the relative path from the metadata and construct a valid absolute path for the image
            img_labels = dict() # Placeholder to contain all the traffic labels for the valid image.

            # Store all the labelled traffic light information from the image.
            for label in metadata.get('boxes'):
              try:
                if label.get('label'):
                  img_labels[label.get('label')] = [label.get('x_max'), label.get('x_min'), label.get('y_max'), label.get('y_min')]
              except:
                print('Problem: ', label)
                print(metadata)
                sys.exit('Mit')
            
            # Store the filepath and label metadata for the valid image into a single datastructure for quick retrieval.
            img_contents['fp'] = val_fp
            img_contents['labels'] = img_labels

            return_obj.append(img_contents)

    if simple_dir or simple_off or simple_label:
        mod_return = list()
        for i in range(len(return_obj)):
            labels = return_obj[i].get('labels')
            unique_ele = list(set(labels.keys()))

            if simple_dir: # Remove all directional labels for traffic lights like "RedLeft"
              if True in ['left' in ele.lower() for ele in unique_ele] or True in ['straight' in ele.lower() for ele in unique_ele] or True in ['right' in ele.lower() for ele in unique_ele]:
                continue

            if simple_off: # Remove all labels of type "off" from the dataset
              if True in ['off' in ele.lower() for ele in unique_ele]:
                continue

            if simple_label: # Remove data that contain multiple label types in the image
              if len(unique_ele) == 1: 
                mod_return.append(return_obj[i])
            else:
              mod_return.append(return_obj[i])

        return mod_return

    return return_obj

def visualize_img(dataset, class_result, display_single=True, get_class=False):
    '''
    param: dataset <arr> - This is the parsed valid images, with the filepath and labels.
    param: class_result <str> - Valid labeled classes from the dataset (ex. Red, Redleft, Green, Yellow, etc.)
    param: display_single <bool> - If true, display a single example of the specified class label. Otherwise display many examples.
    param: get_class <bool> - If you want all the images for a specified class label, you can set this to True and it will return a list of images that contain the specified label.
    return: return_obj <arr> or None - If get_class is True, then the return will be a list of all the images from the dataset that contain the specified label.
    ''' 
    return_obj = list()

    # Validate user class_result
    if class_result.lower() not in ['red', 'green', 'off', 'yellow', 'redleft']:
        raise Exception('Invalid class_result for image visualization: {}'.format(class_result))

    try:
        for image in dataset:
            if class_result in [item.lower() for item in image.get('labels')]:
                if not get_class:
                    img = cv2.imread(image.get('fp')) # Load image data
                    if img is None: continue
                    dim = img.shape

                    print('\nMetadata for the image: {}\n'.format(image.get('fp')))
                    print('Image dimensions: {}'.format(dim))
                    print('Image height: {}'.format(dim[0]))
                    print('Image width: {}'.format(dim[1]))
                    print('Number of channels: {}'.format(dim[2]))
                    print('Displaying image...')

                    plt.imshow(img) # Display the image
                    plt.show()

                    # If the user wants to display a single example of the specified class_result
                    if display_single:
                        return

                else:
                    return_obj.append(image.get('fp'))
    except KeyboardInterrupt:
        sys.exit("Press CTRL-C to terminate continous loop.")

    if get_class:
        print('The {} are in these files: {}', class_result.lower(), return_obj)

    return 

def crop(img_fp, y0, y1, x0, x1, display=False):
    '''
    param: img_fp <str> - Absolute filepath of the image from the valid dataset.
    param: y0, y1, x0, x1 <int> - The coordinates to specify the cropped region.
    param: display <bool> - If true, the cropped image will be displayed to the user for further analysis.
    return: cropped_im <arr> - Cropped version of the image.
    Description: Takes an input image and crops the image to the following
    dimensions. This will be usesful to get rid of the unnecessary portions
    of the traffic light images.
    '''
    # Load image
    img = cv2.imread(img_fp) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV by default reads images in BGR format

    im = np.copy(img) # Store a backup
    cropped_im = im[y0:y1, x0:x1] # Crop the image using numpy method.

    if display:
        plt.imshow(cropped_im)
        plt.show()

    return cropped_im


def visualize_data(dataset):
  '''
  param: dataset <dict> - This is the parsed .yaml contents of the Bosch Traffic light dataset
  return: None
  This will take the parsed dataset and plot the counts of all the valid labels in the dataset
  '''
  data_count = {
      "red": 0,
      "yellow": 0,
      "green": 0,
      "off": 0
  }

  # Loop through the dataset and get a tally of all the labels
  for data in dataset:
    label = list(set(data.get('labels')))[0].lower()
    data_count[label] += 1

  # Plot the label counts
  for label in data_count.keys():
    print("# of {} labeled images in dataset: {}".format(label, data_count[label]))
  plt.bar(range(len(data_count)), list(data_count.values()), align='center')
  plt.xticks(range(len(data_count)), list(data_count.keys()))
  plt.xlabel('Dataset labels', fontsize=12)
  plt.ylabel('# of images with specified labels', fontsize=12)
  plt.show()

  return

def generate_heatmap(dataset):
  '''
  param: dataset <arr> - This is a list representation of all the valid data, where each element is a dict containing
    image filepath, label and annotated bounding box coordinate information.
  return: None
  Given a valid dataset, generate a heat map of all the points on a 1280x720 representative image. These points represent
    bounding boxes that were annotated in the dataset.
  '''
  x, y = list(), list()

  # Collect all the max/min coordinates from the labels of the dataset
  for data in dataset:
    for label in data.get('labels'):
      x_max, x_min, y_max, y_min = data['labels'][label][0], data['labels'][label][1], data['labels'][label][2], data['labels'][label][3]
      x.extend([x_max, x_min])
      y.extend([y_max, y_min])

  # Generate a scatter plot
  plt.scatter(x,y)
  # This is the same coordinate system as the images from the Bosch Dataset
  plt.xlim(0, 1280)
  plt.ylim(720, 0)
  plt.xlabel('Horizontal pixel resolution', fontsize=12)
  plt.ylabel('Vertical pixel resolution', fontsize=12)
  plt.show()
  print('Info to assist with vertical crop:\nThe y-max is {}'.format(max(y)))

  return

print(parse_yaml())



