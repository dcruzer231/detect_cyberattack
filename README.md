# Cruz-Dawson-Summer2021

This repository holds the awesome code that Daniel Cruz wrote for his ORNL internship in the Summer of 2021!  With permission from my mentor Joel Dawson.  Data used is not provided in this repository.

# Daniel Cruz 2021
This repository holds the machine learning model used to train on HASP Images (Harmonically Aligned Signal Projection) of back-channel current readings.

- makeDasp.py houses several methods I made to generate HASP images.  The main functions you would want to mess with
just for generating HASP images are located at the end of the program.  Each method is used for a different dataset
(for example one for the sweep dataset and one for the attack dataset).  Each method has a savepath variable that will point to 
the directory that will save all the images.  createInstructionImages() and createInstructionImages_forRandom() also generate a 
csv that labels the images with instruction parameters.  Note that a required module dasp.py module is not present in this repository as it was not written by me.

- There are three CNN models in three programs: attackcnn.py, instructionCnn.py, mixedinstructionCnn.py.
attackcnn.py is a classifier that train images if they are being cyberattacked or if they are normal.  This should be used when
using the createImages() method in makeDasp.py.  instructionCnn.py is a classifier that trains based on the full instruction.  For example, a single class would be (foward-100-200).  mixedinstructionCnn.py is a mixed cnn model that is both a classifier and a regressor.  It classifies the state of the stepper motor and uses regression to predict the speed of the stepper motor.  Here are some variables in these programs to keep in mind:
    - newModel - set to either retrain and make a save or just run the model on a validation set.
    - save - the name of the folder the program will generate that will hold the save data for the model
    - data_path - point this to the folder with the data.
    - preprocessing_selectChannel - This can be used to toggle on or off specific channels in the 4-channel HASP images.  Simply **uncomment if you want to deactivate that channel**
    - pd.read_csv("instruction.csv") - point this method to the csv file with the labels this is only for the instruction CNN and this csv is made by the instruction methods in makeDasp.
    - seed - Only found in mixedinstructionCnn.py.  This seeds the shuffling of the data in case you want to keep the same shuffle.
    
Only in mixedinstructionCNN, there are optional parameters you can pass to the program to interface with some of these options:
- -s to train and save the model again.  Otherwise it just runs data through the model
- -m to mask any of the 4 channels on the HASP image.  This should be a 4 bit binary value. 1 means the channel will not be deactived, 0 means that channel will be deactived.  Please remember that any masking configuration you train on, should also be the same masking configuration you test on later.

**example**: _python3 mixedinstructionCNN.py -s -m 1101_
This will train the model and make a save and will have all but the 3rd channel in the image active.


The rest of the programs are just simple utility programs used to generate some metric from the data.  

### Data format
##### HASP images
The HASP images are generated png images.  These images are 4 channels (red,green,blue,alpha) were each channel is a 2-dimensional HASP generated from a single current channel.  The labels for each channel have been commented in the makeDasp.py program.  Note that these HASP images will not look very appealing when viewed because the alpha channel will make portions of the image transparent.

##### CSV labels
The csv file with the image labels are designed for the tensorflow flow_from_dataframe() generator.  Note that this is only used for the instruction cnn models.  The csv will have four columns: 
- filename: the name of the image file
- instruction: (FWD, IDLE, REV) Used for classification
- steps: the speed in steps/second. Used for regression.
- number of steps: can be used to calculate the duration of the instruction. Not used in the models




### My work enviroment:
5.8.0-49-generic Linux UBUNTU Desktop OS\
I trained using the tensorflow container with nvdia-docker version: Docker version 19.03.13, build 4484c46d9d\
I used a personal modified version of tensorflow/tensorflow:latest-gpu container with tensorflow 2.0.  I imagine you can run these programs just fine without the docker.
python version was python3 3.8.10