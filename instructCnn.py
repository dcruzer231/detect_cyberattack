"""
written by Daniel Cruz
The cnn program used to learn instruction from dasp images
"""


#enable this to make a new model based on the data_path
#otherwise use the model saved under "save"
newModel = True
save = "./instructionSave"

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd

data_path = './Data/images/instruction/'

img_height = 256 #original size 1600
img_width = 256 #original size 1200

batch_size=16//2

classDir = ["good","bad"]

def preprocessing_selectChannel(image):
  """
  'stepper_driver'
  'ps_wall'
  'driver_ps'
  'instchan'
  """

  image[:,:,0].fill(0)
  # image[:,:,1].fill(0) 
  image[:,:,2].fill(0) 
  image[:,:,3].fill(0)
  return image



df = pd.read_csv("instruction.csv")
print(df)
df["full_instruction"] = df["instruction"] + df["steps"].astype(str) + df["number of steps"].astype(str)
print(pd.unique(df["full_instruction"]))


train_df = df.sample(frac=0.6)


test_df = df.drop(train_df.index)
val_df = test_df.sample(frac=0.5)
test_df = test_df.drop(val_df.index)
print(train_df)
print(test_df)
print(val_df)


train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing_selectChannel)


train_generator = train_datagen.flow_from_dataframe(train_df,directory=data_path,x_col="filename",y_col="full_instruction",target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, color_mode="rgba")
validation_generator = train_datagen.flow_from_dataframe(test_df,directory=data_path,x_col="filename",y_col="full_instruction",target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, color_mode="rgba")

#used for building the confusion matrix.
test_generator = train_datagen.flow_from_dataframe(val_df,directory=data_path,x_col="filename",y_col="full_instruction",target_size=(img_height,img_width), batch_size=batch_size, shuffle=False, color_mode="rgba")

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
import distutils

def create_model():

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(img_height,img_width,4), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
 
  model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
  
  model.add(tf.keras.layers.Flatten())
  
  model.add(tf.keras.layers.Dense(64,activation ='relu'))
  model.add(tf.keras.layers.Dense(8,activation = 'softmax'))
  
  return model


if newModel:
  model = create_model()
  model.summary()

  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  history = model.fit(train_generator,  epochs=1, validation_data=validation_generator, validation_steps = batch_size)
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