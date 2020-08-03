import yaml
import os
import sys

BASEDIR = os.getcwd().split('helper')[0] # Remove the helper directory from the project base filepath
YAML_FP = BASEDIR + "data/dataset_train_rgb/train.yaml"
UPDATE_BASE_FP = BASEDIR + "data/dataset_train_rgb/rgb/test"

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
            path = UPDATE_BASE_FP + "/{}".format(update_file)
            content = dict()
            content['boxes'] = list()
            content['path'] = path
            yaml_contents.append(content)

        write_result = yaml.dump(yaml_contents, write_file)

    write_file.close()

    return
