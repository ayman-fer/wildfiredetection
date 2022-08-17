#!/usr/bin/env python
# coding: utf-8

# # EfficientNet Implementation   - EfficientNet- B0
# 
# ## Training a Custom  Model from scratch

# ![image.png](attachment:image.png)

# # Data Pre Processing

# In[1]:


import numpy as np
import tensorflow as tf

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

dataset_path = os.listdir('frames/Test1')

print (dataset_path)  #what kinds of classes are in this dataset

print("Types of classes labels found: ", len(dataset_path))


# In[2]:


class_labels = []

for item in dataset_path:
 # Get all the file names
 all_classes = os.listdir('frames/Test1' + '/' +item)
 #print(all_classes)

 # Add them to the list
 for room in all_classes:
    class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))
    #print(class_labels[:5])


# In[3]:



# Build a dataframe        
df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
print(df.head())
print(df.tail())


# In[4]:


# Let's check how many samples for each category are present
print("Total number of images in the dataset: ", len(df))

label_count = df['Labels'].value_counts()
print(label_count)


# In[5]:


import cv2
path = 'frames/Test1/'
dataset_path = os.listdir('frames/Test1')

im_size = 260

images = []
labels = []

for i in dataset_path:
    data_path = path + str(i)  
    filenames = [i for i in os.listdir(data_path) ]
   
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)


# In[6]:



#This model takes input images of shape (224, 224, 3), and the input data should range [0, 255]. 

images = np.array(images)

images = images.astype('float32') / 255.0
images.shape


# In[7]:


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
y=df['Labels'].values
print(y)

y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)


# In[8]:



y=y.reshape(-1,1)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y) #.toarray()
print(Y[:5])
print(Y[35:])


# In[9]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


images, Y = shuffle(images, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.2, random_state=415)
#from sklearn.model_selection import train_test_split
import numpy as np
#from sklearn.model_selection import KFold

# kf = KFold(n_splits=8)
# images, Y = shuffle(images, Y, random_state=1)
# for train_index, test_index in kf.split(images):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     train_x, test_x = images[train_index], images[test_index]
#     train_y, test_y = Y[train_index], Y[test_index]

#inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# 
# # EfficientNet Implementation :
# 
# 

# In[10]:


from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB2

NUM_CLASSES = 2
IMG_SIZE = 260
size = (IMG_SIZE, IMG_SIZE)


inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))


# Using model without transfer learning

outputs = EfficientNetB2(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)


# In[11]:
#%load_ext tensorboard
import tensorflow as tf
import datetime
#!rm -rf /logs/
# log_dir = "logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('training.log', separator=',', append=False)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"] )

model.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint("EfficientNetB2.pb", save_best_only=True)
early_stopper = tf.keras.callbacks.EarlyStopping(patience=5)
hist = model.fit(train_x, train_y, epochs=30, verbose=2, callbacks=[csv_logger])

#tensorboard --logdir logs/fit

# In[12]:


import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.show
    plt.savefig("Output/plots.pdf")


plot_hist(hist)


# In[13]:


preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# # Testing Efficient Model On Unseen data

# In[14]:


from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input


img_path = 'frames/Training/resized_test_fire_frame0.jpg'

#img = image.load_img(img_path, target_size=(224, 224))
#x = img.img_to_array(img)

img = cv2.imread(img_path)
img = cv2.resize(img, (260, 260))

x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

print('Input image shape:', x.shape)

my_image = imread(img_path)
imshow(my_image)


# In[15]:


preds=model.predict(x)
preds     # probabilities for being in each of the 3 classes


# In[18]:



# Cuda and cudnn is installed for this tensorflow version. So we can see GPU is enabled
tf.config.experimental.list_physical_devices()


# In[21]:


get_ipython().run_cell_magic('timeit', '-n1 -r1 ', "with tf.device('/CPU:0'):\n    cpu_performance =model.fit(train_x, train_y, epochs=30, verbose=2)\n    cpu_performance")


# In[22]:



get_ipython().run_cell_magic('timeit', '-n1 -r1 ', "with tf.device('/GPU:0'):\n    gpu_performance =model.fit(train_x, train_y, epochs=30, verbose=2)\n    gpu_performance")


# In[ ]:


# CPU completed the training in 7 min 53 Seconds and GPU did that training in 25.6 seconds

