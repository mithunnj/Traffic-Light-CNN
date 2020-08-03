# APS360-Traffic-Light-Classifier
APS 360 final project: Traffic Light Classifier for Self-Driving Applications

| Name (URL - LinkedIn) | Email | 
|:-------------:|:------------:| 
| [Mithun Jothiravi](https://www.linkedin.com/in/mithunjothiravi/)    | mithun.jothiravi@mail.utoronto.ca |
| [Graeme Aylward](https://www.linkedin.com/in/graeme-aylward-422a35181/)      | graeme.aylward@mail.utoronto.ca      | 
| [Caitlin Everett](https://www.linkedin.com/in/caitlin-everett/) | caitlin.everett@mail.utoronto.ca    |
| [Maxwell Gyimah](https://www.linkedin.com/in/maxwell-gyimah-53476715a/) | maxwell.gyimah@mail.utoronto.ca   |


## Getting started

### Dataset
The dataset that will be used for this project will be the [Bosch Small Traffic Lights dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset). To follow this project locally, go to the provided URL, and request a download link of the dataset under the **Dataset Download** heading. Due to Github's file size limits, the contents of the data folder will be empty in this repo, and will needed to be downloaded manually.

**NOTE: Read the following steps carefully**
1. The data will come into split `.zip` files in the form: `.zip.001, .zip.002, etc.` You have to follow the steps outlined [here](https://hiro.bsd.uchicago.edu/node/3168) to extract the file properly.
2. The project consists of code that depends on a specific location of all the data files. So after you follow the steps above, extract the files in the following directory: `{Project Root}/data`, where **Project Root** is where you `git clone` this project onto your system.
3. The data is split into **train** and **test** versions, and there is a **RGB** and **RII** version. You can read on the [Bosch Small Traffic Lights dataset website](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) about the differences in the colour formats, but at this stage all you need to download are all the **training RGB** datasets (ex. `dataset_train_rgb.zip.001`).

### Setting up your machine
- It's best to do your Python development in an environment. So follow the steps here to create an Anaconda environment for this project with Python v.3.6 - instructions [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/).
- Git clone this project into your local repo. Ensure that you Git is configured properly on your machine so that you can contribute to the project.
- Install all the project requirements as outlined in the `requirements.txt` file by running the following, in your conda environment: `conda install --file requirements.txt`

### Project structure
- process_data.py: This file contains helper functions to process the raw data (.yaml & images).
- baseline_model.py: This file contains the contents of the project's first milestone.

It's best practice to have your main script (ex. `baseline_model.py`) clean and straight forward, and have all your helper functions in a seperate helper file (ex. `process_data_helper.py`)

### Additional info
- All the labels are described in a `.yaml` file, which can be found in the Bosch Traffic Light dataset that you downloaded. This file contains a filepath for the images, and the annotated traffic light colours. But not all the images have annotated labels for the traffic light colours. So in the processing pipeline in `baseline_model.py`, the first step I made was to read the `.yaml` file and return only the images that have annotated labels. 
    - There are multiple label options like: Redleft, Red, off, Green, Yellow etc. Using the filtering process described above, we can simplify our project to ignore the labels that are out of the scope of the project like Redleft for example.

- I created a helper function to help visualize some of the images (`visualize_img`). Read the function description to understand it's entire utility, but it will help you understand what image you're looking at during your development.

- This was my approach: 
    -  To remove unecessary information, I cropped the image vertically such that we only keep 0px to 575px. The reason is, I figured from looking at most of the pictures that traffic lights are mostly in the top portions of pictures, so I wanted to remove all the road, car, lane line, etc information that would be irrelevant to our pipeline. 
    - I noticed that the traffic lights in the image have a black rectangular shape. So the cropped images are passed into a rectangle detection function called `detect_rectangle`. This function still needs to be fine tuned to detect traffic lights properly. The coordinates of possible traffic light candidates from the image are noted and the original image is cropped, so that we disregard all the useless portions of the image and only focus on the possible traffic light candidate. 
    - The possible candidates are fed into another HSV colour detection pipeline called `hsv_processing`. This function is still in progress, but the idea is: HSV is easier to deal with than RGB because only one colour channel representes the colour in HSV compared to RGB where a single colour is a combination of all three channels (you can mess around with an HSV colour wheel [here](https://alloyui.com/examples/color-picker/hsv.html)). This function also needs to be tuned better to handle the variations of hue values to represent red, yellow and green respectively.

### Project progress
- **Last updated:** Mit - July 4, 2020 
- Created a data processing pipeline that crops the bottom portion of the image, and passes the image into an image processing pipeline that attempts to detect rectangles, and detect colours from the possible traffic light candidates as described above.
- This was a direction that I thought could work, but I am open to other possible avenues.