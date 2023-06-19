"""
written by Daniel Cruz
The main cnn program used to learn HASP program.
"""


#enable this to make a new model based on the data_path
#otherwise use the model saved under "save"
newModel = True
save = "./allChSave"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import distutils


def preprocessing_selectChannel(image):
  """
  'stepper_driver'
  'ps_wall'
  'driver_ps'
  'instchan'
  """

  #uncomment to apply masking
  # image[:,:,0].fill(0)
  # image[:,:,1].fill(0) 
  # image[:,:,2].fill(0) 
  # image[:,:,3].fill(0)
  return image

data_path = './Data/images/fc_200_bw_500/'

img_height = 256 #original size 1600
img_width = 256 #original size 1200

batch_size=16//2

classDir = ["good","bad"]

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,preprocessing_function=preprocessing_selectChannel)

seed = 10
train_generator = train_datagen.flow_from_directory(data_path,target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, subset="training",classes=classDir,color_mode="rgba")
validation_generator = train_datagen.flow_from_directory(data_path,target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, subset="validation",classes=classDir,color_mode="rgba")

#used for building the confusion matrix.
test_generator = train_datagen.flow_from_directory(data_path,target_size=(img_height,img_width), batch_size=batch_size, shuffle=False, subset="validation",classes=classDir,color_mode="rgba")


def create_model():

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(img_height,img_width,4), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
 
  model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
  
  model.add(tf.keras.layers.Flatten())
  
  model.add(tf.keras.layers.Dense(64,activation ='relu'))
  model.add(tf.keras.layers.Dense(2,activation = 'softmax'))
  
  return model


if __name__ == '__main__':
  
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