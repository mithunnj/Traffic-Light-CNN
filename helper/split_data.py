'''
Description: This helper function will be used to split the dataset into Test, Train and Validation subsets.
'''

import os
import sys

def split_dataset(CROP_IMGS):
    '''
    Take all the detected objects from the original Bosch Dataset (the output from the YOLOv3 portion of the pipeline) and
    split the data into a training set (60% of the dataset) and validation set (40% of the dataset).
    All the new holdout test set will be contained in it's own directory.
    '''
    file_count = 0

    data_split = dict()
    data_split['train'] = list()
    data_split['val'] = list()
    data_split['test'] = list()

    directories = os.listdir(CROP_IMGS)

    for directory in directories:
        if ".DS_Store" in directory or "IMG_" in directory: # This is a corner case folder that is generated
            continue

        sub_files = os.listdir('{}/{}'.format(CROP_IMGS, directory))
        sub_files.remove("file.name")

        for sub_file in sub_files:
            if file_count < 17521: # ~60% of the total object detected dataset (29201 objects) for the Train Dataset
                file_count += 1
                data_split['train'].append("/{}/{}".format(directory, sub_file))
            elif 17521 <= file_count <= 23360: # ~20% of the total object detected dataset for the Validation Dataset
                file_count += 1
                data_split['val'].append("/{}/{}".format(directory, sub_file))
            elif 23360 < file_count: # ~20% of the total object detected dataset for the holdout test dataset
                file_count += 1
                data_split['test'].append("/{}/{}".format(directory, sub_file))

    #print("Length of train set: {}".format(len(data_split['train'])))
    #print("Length of validation set: {}".format(len(data_split['val'])))
    #print("Length of test set: {}".format(len(data_split['test'])))


    return data_split
