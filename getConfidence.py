"""
Written by Daniel Cruz
This program is used to obtain the csv of a prediction that compares
Correct prediction, distance from 0, confidence
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import distutils
import h5py

from makeDasp import getRunHASP

from attackcnn import preprocessing_selectChannel, create_model

save = "./allChSave"

titles = ["index","stepper_driver", "ps_wall","driver_ps","instchan","run_gt","inst_gt"]


sweepPath = "./Data/frequency_sweep/"
attackPath = "./Data/simulated_attack/known-bad/"
goodPath = "./Data/simulated_attack/baseline-known-good/"

savePath = "./Data/images/tests/"

sweepFiles = ["data-HALF-BLUE-LAB-r10000d1800-091820-182807.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-142849.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-154000.h5f","data-HALF-BLUE-LAB-r10000d3600-092120-170842.h5f"]
goodFiles = ["data-increment-BL-r10000d18000-051320-125438.h5f","data-increment-BL-r10000d18000-051320-184456.h5f","data-increment-BL-r10000d18000-051420-110816.h5f","data-increment-BL-r10000d18000-051420-171038.h5f"]
attackFiles = ["data-increment-MUTSPD5-r10000d18000-061520-105843.h5f","data-increment-MUTSPD5-r10000d18000-061620-002749.h5f","data-increment-MUTSPD-r10000d18000-051720-150804.h5f","data-increment-MUTSPD-r10000d18000-051820-114805.h5f","data-increment-MUTSPD-r10000d18000-052920-153932.h5f"]
    

#enable this to make a new model based on the data_path
#otherwise use the model saved under "save"
newModel = False
save = "./allChSave"


data_path = './Data/images/fc_200_bw_500/'

img_height = 256 #original size 1600
img_width = 256 #original size 1200

batch_size=16//2

classDir = ["good","bad"]

# def preprocessing_selectChannel(image):
#   """
#   'stepper_driver'
#   'ps_wall'
#   'driver_ps'
#   'instchan'
#   """

#   image[:,:,0].fill(0)
#   image[:,:,1].fill(0) 
#   # image[:,:,2].fill(0) 
#   image[:,:,3].fill(0)
#   return image


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,preprocessing_function=preprocessing_selectChannel)


train_generator = train_datagen.flow_from_directory(data_path,target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, subset="training",classes=classDir,color_mode="rgba")
validation_generator = train_datagen.flow_from_directory(data_path,target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, subset="validation",classes=classDir,color_mode="rgba")

#used for building the confusion matrix.
test_generator = train_datagen.flow_from_directory(data_path,target_size=(img_height,img_width), batch_size=batch_size, shuffle=False, subset="validation",classes=classDir,color_mode="rgba")


# def create_model():

#   model = tf.keras.models.Sequential()
#   model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(img_height,img_width,4), activation='relu'))
#   model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
 
#   model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
#   model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
  
#   model.add(tf.keras.layers.Flatten())
  
#   model.add(tf.keras.layers.Dense(64,activation ='relu'))
#   model.add(tf.keras.layers.Dense(2,activation = 'softmax'))
  
#   return model

if newModel:
  model = create_model()
  model.summary()

  class_weight = {0: 0.9,1: 1.12}

  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  history = model.fit(train_generator,  epochs=1, validation_data=validation_generator, validation_steps = batch_size, class_weight=class_weight)
  model.save(save)

else: 
  model = tf.keras.models.load_model(save)

fileNames = test_generator.filenames
print(len(fileNames))
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(test_generator,batch_size+1)

print(Y_pred[1,:].min())
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

print(classification_report(test_generator.classes, y_pred))

with open("confidence.csv", "w") as fout:
  fout.write("name,correct,confidence,distance\n")
  for i,name in enumerate(fileNames):
    fname,ind,distance = name.split("_")
    #remote the .png from distance
    distance = distance[:-4]
    correct = ""
    if y_pred[i] == test_generator.classes[i]: correct = "1"  
    else: correct = "0" 
    # print(fname,correct,Y_pred[i].max(),distance)
    fout.write(fname+","+correct+","+str(Y_pred[i].max())+","+str(distance)+"\n")


#this prints all pictures that are wrong.
# wrongPred = (y_pred != test_generator.classes)
# print(wrongPred.shape)
# ind = 0
# for boolean in wrongPred:
#   if(boolean == True):
#       print(ind, y_pred[ind], fileNames[ind])
#   ind += 1

if newModel:
  fig, ax = plt.subplots()
  ax.plot(history.history['accuracy'],label = 'train')
  ax.plot(history.history['val_accuracy'],label = 'test')
  ax.set_title('Accuracy')
  ax.legend(loc='lower right')
  plt.savefig("./lossGraph.png")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'],label = 'train')
  ax.plot(history.history['val_loss'],label = 'test')
  ax.set_title('Loss')
  ax.legend(loc='upper right')
  plt.savefig("./accuracyGraph.png")