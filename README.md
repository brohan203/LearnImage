# LearnImage
Tool for stain segmentation of dark-field immunohistochemistry images.

This code was produced by Rohan Borkar in collaboration with Thierno Madjou Bah and Nabil Alkayed at the Oregon Health and Sciences University. This script runs on any 3-channel images and can recognize any combination of colors or channels so whether you have lots of channels and must detect different combinations, or single channel images this script can find percent coverage.

The following example depicts an input file of neuronal staining and the result.

<br>

<img src="https://raw.github.com/brohan203/LearnImage/master/input_sample.png" width="332" height="416">  <img src="https://raw.github.com/brohan203/LearnImage/master/output_sample.png" width="332" height="416">

<br>

Instructions: There are two steps to using this code:
    1. Learning the images using machine learning
    2. Running the images using a trained model

For learning images, start out by setting the "training" variable equal to True, specifying the training image by changing the "training_image" to the appropriate file, and marker name. Then run the script within terminal or command prompt. The program will display two windows in succession. In the first window, select points that should be excluded from segmentation such as background and non-significant tissue. In the second window, select points that should be included for segmentation. The training data will be saved based on what marker name was specified. If you are using one marker for different training sets or runs, append a number or other indicator to the end each time you train.

For running images on a trained model, specify the "data_folder" variable to the path to your input folder, set the "output_folder" variable to a non-existing folder name that the output images will be saved to, specify the name of the CSV file data will be stored to, and specify the marker name for training data. Then simply run the program and it will run through the entire data folder using the trained model.
