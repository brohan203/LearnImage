# LearnImage
Tool for stain segmentation of dark-field immunohistochemistry images.

This code was produced by Rohan Borkar in collaboration with Thierno Madjou Bah and Nabil Alkayed at the Oregon Health and Sciences University. This script uses Support Vector Machines to learn dark-field immunohistochemistry image data based on user input. It runs on any 3-channel images and can recognize any combination of colors and some level of texture. Whether you have multiple channels and must detect different combinations of them, or single channel images this script can be used to find percent coverage.

The following example depicts an input file of neuronal staining and the result.

<br>

<img src="https://raw.github.com/brohan203/LearnImage/master/input_sample.png" width="332" height="416">  <img src="https://raw.github.com/brohan203/LearnImage/master/output_sample.png" width="332" height="416">

<br>

Instructions: 

There are two steps to using this code:
    1. Learning the images using machine learning
    2. Running the images using a trained model

For learning images, start out by setting the "training" variable equal to True, specifying the training image by changing the "training_image" to the appropriate file, and marker name. Then run the script within terminal or command prompt. The program will display two windows in succession. In the first window, select points that should be excluded from segmentation such as background and non-significant tissue. In the second window, select points that should be included for segmentation. The training data will be saved based on what marker name was specified. If you are using one marker for different training sets or runs, append a number or other indicator to the end each time you train.

For running images on a trained model, specify the "data_folder" variable to the path to your input folder, set the "output_folder" variable to a non-existing folder name that the output images will be saved to, specify the name of the CSV file data will be stored to, and specify the marker name for training data. Then simply run the program and it will run through the entire data folder using the trained model.


Citation:
We request anyone who uses this code to kindly acknowledge the authors by citing the following paper of ours that will be submitted soon. A link to the paper will be provided after publication.

Soluble epoxide hydrolase inhibition improves cognition in a mouse model of vascular cognitive impairment (Under preparation).
Thierno M. Bah, Ph.D.; Catherine M. Davis, Ph.D.; Elyse Allen, B.Sc.; Rohan N. Borkar, B.Sc.; Marjorie R. Grafe, MD, Ph.D.; Ruby Perez, B.Sc; Jacob Raber, Ph.D.; Martin M. Pike, Ph.D.; Nabil J. Alkayed, M.D., Ph.D.
