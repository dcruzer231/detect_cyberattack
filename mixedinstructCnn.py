"""
written by Daniel Cruz
The cnn program used to learn instruction from dasp images
This program uses a mixed output model
"""

import sys


#enable this to make a new model based on the data_path
#otherwise use the model saved under "save"
newModel = False
mask = [1,1,1,1]
if len(sys.argv) > 1:
  if "-s" in sys.argv:
    newModel = True 
  if "-m" in sys.argv:
    i = sys.argv.index("-m")
    mask = [bit for bit in sys.argv[i+1]]
    print(mask)

save = "./mixedSave"

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

data_path = './Data/images/instruction/'


img_height = 256 #original size 1600
img_width = 256 #original size 1200

batch_size=16//2



#uncomment a line to clear that channel,
#used to train on single channels or certain
#combinations of channels
label = ""
def preprocessing_selectChannel(image):
  

  """
  'stepper_driver_i',
  'stepper_driver_v',
  'driver_ps',
  'instchan_v',
  """
  #uncomment this to manually apply masking

  # image[:,:,0].fill(0)`
  # image[:,:,1].fill(0)     `
  # image[:,:,2].fill(0)     
  # image[:,:,3].fill(0)

  if mask[0] == 0:
    image[:,:,0].fill(0)
  if mask[1] == 0:
    image[:,:,1].fill(0)
  if mask[2] == 0:     
    image[:,:,2].fill(0) 
  if mask[3] == 0:    
    image[:,:,3].fill(0)
  return image


#Other csv files used

# df = pd.read_csv("instruction_data-increment-BL-r10000d18000-051320-125438.h5f.csv")
# df = pd.read_csv("instruction.csv")
df = pd.read_csv("instruction_data-random-inst-r10000d7200-070821-131756.h5f.csv")
# df = pd.read_csv("instruction_data-eval-rand-r40000d7200-072121-134715.h5f.csv")


LabelBinarizer = LabelBinarizer()
d= LabelBinarizer.fit_transform(df["instruction"])

df["instruction"] = df["instruction"].astype(object)
for i in range(d.shape[0]):
  df["instruction"][i] = d[i].tolist()

df["steps"] = df["steps"].astype(float)
df["number of steps"] = df["number of steps"].astype(float)


steps_mean = df["steps"].mean()
steps_std = df["steps"].std(ddof=0)
df["steps"] = (df["steps"] - steps_mean)/steps_std

num_steps_mean = df["number of steps"].mean()
num_steps_std = df["number of steps"].std(ddof=0)
df["number of steps"] = (df["number of steps"] - num_steps_mean)/num_steps_std

print(d.dtype)
print(df.head())
print(df.dtypes)
print(df["instruction"][1][1])


seed = None

train_df = df.sample(frac=0.8, random_state=seed)


# train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
val_df = test_df.sample(frac=0.5, random_state=seed)
test_df = test_df.drop(val_df.index)
print(train_df)
print(test_df)
print(val_df)


train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocessing_selectChannel)

# y_cols = ["instruction", "steps", "number of steps"]
y_cols = ["instruction", "steps"]


train_generator = train_datagen.flow_from_dataframe(train_df,directory=data_path,class_mode="multi_output",x_col="filename",y_col=y_cols,target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, color_mode="rgba")
validation_generator = train_datagen.flow_from_dataframe(test_df,directory=data_path,class_mode="multi_output",x_col="filename",y_col=y_cols,target_size=(img_height,img_width), batch_size=batch_size, shuffle=True, color_mode="rgba")

#used for building the confusion matrix.
test_generator = train_datagen.flow_from_dataframe(val_df,directory=data_path,class_mode="multi_output",x_col="filename",y_col=y_cols,target_size=(img_height,img_width), batch_size=batch_size, shuffle=False, color_mode="rgba")

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

  inp = Input((img_height,img_width,4))
  x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inp)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
 
  x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')(x)
  
  x = tf.keras.layers.Flatten()(x)
  
  x = tf.keras.layers.Dense(128,activation ='relu')(x)
  x = tf.keras.layers.Dense(128,activation ='relu')(x)
  x = tf.keras.layers.Dense(64,activation ='relu')(x)

  # x = tf.keras.layers.Dense(32,activation ='relu')(x)

  out1 = tf.keras.layers.Dense(3,activation = 'softmax',name="instruction")(x)
  out2 = tf.keras.layers.Dense(1,activation='linear',name="steps")(x)
  # out3 = tf.keras.layers.Dense(1,activation='linear',name="numberOfSteps")(x)


  
  # return Model(inp,[out1,out2,out3])
  return Model(inp,[out1,out2])



if newModel:
  model = create_model()
  # model.summary()

  # model.compile(loss={"instruction":"categorical_crossentropy","steps":"mse","numberOfSteps":"mse"}, optimizer="adam", metrics={"instruction":"accuracy","steps":"mean_absolute_error","numberOfSteps":"mean_absolute_error"})
  model.compile(loss={"instruction":"categorical_crossentropy","steps":"mse"}, optimizer="adam", metrics={"instruction":"accuracy","steps":"mean_absolute_error"})

  history = model.fit(train_generator,  epochs=4, validation_data=validation_generator, validation_steps = batch_size)
  model.save(save)

else: 
  model = tf.keras.models.load_model(save)

fileNames = test_generator.filenames
print(len(fileNames))
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

# for x,y in test_generator:
#   print(x.shape)
#   print(len(y))
#   # break
# Y_pred1,y_pred2, y_pred3 = model.predict(test_generator,batch_size+1)
Y_pred1,y_pred2 = model.predict(test_generator,batch_size+1)


labels = np.array(val_df["instruction"].tolist())
steps_true = np.array(val_df["steps"].tolist())
num_steps_true = np.array(val_df["number of steps"].tolist())


y_pred1 = np.argmax(Y_pred1, axis=1)
y_true1 = np.argmax(labels, axis=1)

print('Confusion Matrix')

print(confusion_matrix(y_true1, y_pred1))

print(classification_report(y_true1, y_pred1))



#convert back to raw value from z score
steps_true_raw = steps_mean+steps_true*steps_std
y_pred2_raw = steps_mean+y_pred2*steps_std

num_steps_true_raw = num_steps_mean+num_steps_true*num_steps_std
# y_pred3_raw = num_steps_mean+y_pred3*num_steps_std

print("mean squared error of steps: \t\t",mean_squared_error(steps_true_raw, y_pred2_raw))
print("root mean squared error of steps: \t",mean_squared_error(steps_true_raw, y_pred2_raw, squared=False))
print("mean absolute error of steps: \t\t",mean_absolute_error(steps_true_raw, y_pred2_raw))



# print("mean squared error of number of steps: \t",mean_squared_error(num_steps_true_raw, y_pred3_raw))

plotsavePath = "./Data/visualization/"
fig, ax = plt.subplots()
ax.set_xlabel("True Values")
ax.set_ylabel("Prediction")
ax.set_title("steps")
ax.plot(steps_true_raw,y_pred2_raw, "o")

#get the minimum of either groundtruth or prediction and the max of either groundtruth or prediction
identityRange = [min(min(steps_true_raw),min(y_pred2_raw)), max(max(steps_true_raw),max(y_pred2_raw))]
#plot identity line
ax.plot(identityRange,identityRange)
plt.savefig(plotsavePath+label+"regression")

# fig, ax = plt.subplots()
# ax.set_xlabel("True Values")
# ax.set_ylabel("Prediction")
# ax.set_title("number of steps")
# ax.plot(num_steps_true_raw,y_pred3_raw, "o")
#get the minimum of either groundtruth or prediction and the max of either groundtruth or prediction
# identityRange = [min(min(num_steps_true_raw),min(y_pred3_raw)), max(max(num_steps_true_raw),max(y_pred3_raw))]
#plot identity line
# ax.plot(identityRange,identityRange)

wrongl = y_pred1 != y_true1
correctl = y_pred1 == y_true1

fig, ax = plt.subplots()
plt.sca(ax)
plt.xlabel("Classification")
plt.ylabel("Confidence")
plt.title("Classifier Confidence")

#plot by correct/wrong colour coded
# ax.plot(y_pred1[correctl],np.max(Y_pred1[correctl], axis=1), "o")
# ax.plot(y_pred1[wrongl],np.max(Y_pred1[wrongl], axis=1), "o",color="r")

#plot by color coded classes
ax.plot(y_pred1[y_true1 == 0],np.max(Y_pred1[y_true1 == 0], axis=1), "o",color="r", label="fwd")
ax.plot(y_pred1[y_true1 == 1],np.max(Y_pred1[y_true1 == 1], axis=1), "o",color="g", label="idle")
ax.plot(y_pred1[y_true1 == 2],np.max(Y_pred1[y_true1 == 2], axis=1), "o",color="b", label="rev")
ax.legend()



#convert x axis into labels
plt.xticks([0,1,2],LabelBinarizer.inverse_transform(np.array([[1,0,0],[0,1,0],[0,0,1]])))
# plt.ylim(bottom=0.9)
plt.savefig(plotsavePath+label+"classification")


plt.show()



if newModel:
  # fig, ax = plt.subplots()
  # ax.plot(history.history['accuracy'],label = 'train')
  # ax.plot(history.history['val_accuracy'],label = 'test')
  # ax.set_title('Accuracy')
  # ax.legend(loc='lower right')
  plt.savefig("./lossGraph.png")

  fig, ax = plt.subplots()
  ax.plot(history.history['loss'],label = 'train')
  ax.plot(history.history['val_loss'],label = 'test')
  ax.set_title('Loss')
  ax.legend(loc='upper right')
  # plt.savefig("./accuracyGraph.png")