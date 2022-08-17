#!/usr/bin/env python
# coding: utf-8

# ## **TransU-Net**
# 
# 
# 

# In[1]:


import numpy as np
from glob import glob


# In[2]:


from tensorflow import keras


# This example requires `keras-unet-collection`:
# ```
# pip install keras-unet-collection
# ```

# In[8]:





# In[10]:





# In[14]:


from keras_unet_collection import models, utils


# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


# file path after data extraction
path_trainval_img = "frames/Segmentation/Data/Images/"
path_trainval_mask = "frames/Segmentation/Data/Masks/"


# In[ ]:


trainval_input_names = np.array(sorted(glob(path_trainval_img+'*.jpg')))
trainval_label_names = np.array(sorted(glob(path_trainval_mask+'*.png')))


# Training-validation data split
# 
# 

# In[ ]:


L = len(trainval_input_names)
ind_all = utils.shuffle_ind(L)

L_train = int(0.9*L); L_valid = L - L_train
ind_train = ind_all[:L_train]; ind_valid = ind_all[L_train:]

train_input_names = trainval_input_names[ind_train]
train_label_names = trainval_label_names[ind_train]
valid_input_names = trainval_input_names[ind_valid]
valid_label_names = trainval_label_names[ind_valid]

print("Training:validation = {}:{}".format(L_train, L_valid))


# ### Exploratory data analysis

# In[ ]:


#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#def ax_decorate_box(ax):
#    [j.set_linewidth(0) for j in ax.spines.values()]
#    ax.tick_params(axis="both", which="both", bottom=False, top=False,                labelbottom=False, left=False, right=False, labelleft=False)
#    return ax


# In[ ]:


# i_max = 10 # explore 10 images
# input_example = utils.image_to_array(train_input_names[:i_max], size=512, channel=3)
# input_example.shape
# label_example = utils.image_to_array(train_label_names[:i_max], size=512, channel=1)


# In[ ]:


from tqdm import tqdm
import tensorflow as tf
import os
import random
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


# In[ ]:


# i_example = 2

# fig, AX = plt.subplots(1, 2, figsize=(7, 3))
# plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)

# for ax in AX:
#     ax = ax_decorate_box(ax)
    
# AX[0].pcolormesh(np.mean(input_example[i_example, ...], axis=-1), cmap= plt.cm.gray)
# AX[1].pcolormesh(label_example[i_example, ..., 0]>0, cmap=plt.cm.binary_r)
# AX[0].set_title("Original", fontsize=14);
# AX[1].set_title("Segmentation mask", fontsize=14);


# In[ ]:


from keras_unet_collection import models


# TransU-Net pretrained with VGG16 backbone

# In[ ]:


model = models.transunet_2d((512, 512, 3), filter_num=[16, 32, 64, 128, 256], n_labels= 2, stack_num_down=2, stack_num_up=2,
                 embed_dim=768, num_mlp = 3072, num_heads=12, num_transformer=12,
                 activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid', batch_norm=False, pool=True, unpool=True, 
                 backbone='VGG16', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='transunet')


# The second layer of the configured model, i.e., right after an input layer, is expected to be the VGG16 backbone.

# In[ ]:


model.layers[1].name


# For simplicity, this segmentation model is trained with cross-entropy loss with SGD optimizer and a learning rate of 1E-2.

# In[ ]:


METRICS = [
    tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

model.compile(loss=tf.keras.metrics.binary_crossentropy, optimizer=keras.optimizers.SGD(learning_rate=1e-2), metrics= METRICS)


# ..................
# 

# ## **Training**
# 
# with 60 epoches with early stopping  Each epoch containts 10 batches and each batch contains 2 samples.

# def target_data_process(target_array):
#     '''Converting tri-mask of {1, 2, 3} to three categories.'''
#     return keras.utils.to_categorical(target_array-1)

# In[ ]:


def input_data_process(input_array):
    '''converting pixel vales to [0, 1]'''
    return input_array/255.

def target_data_process(target_array):
    target_array[target_array>0]=1  
    return keras.utils.to_categorical(target_array, num_classes=2)


# In[ ]:


valid_input = input_data_process(utils.image_to_array(valid_input_names, size=512, channel=3))
valid_label = target_data_process(utils.image_to_array(valid_label_names, size=512, channel=1))


# In[ ]:


N_epoch = 2 # number of epoches
N_batch = 10 # number of batches per epoch
N_sample = 2 # number of samples per batch

tol = 0 # current early stopping patience
max_tol = 3 # the max-allowed early stopping patience
min_del = 0 # the lowest acceptable loss value reduction 

# loop over epoches
for epoch in range(N_epoch):    
    # initial loss record
    if epoch == 0:
        y_pred =model.predict([valid_input])
        record = np.mean(keras.losses.categorical_crossentropy(valid_label, y_pred))
        print('\tInitial loss = {}'.format(record))
    
    # loop over batches
    for step in range(N_batch):
        # selecting smaples for the current batch
        ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]
        
        # batch data formation
        ## augmentation is not applied
        train_input = input_data_process(utils.image_to_array(train_input_names[ind_train_shuffle], size=512, channel=3))
        train_label = target_data_process(utils.image_to_array(train_label_names[ind_train_shuffle], size=512, channel=1))
        
        # train on batch
        loss_ = model.train_on_batch([train_input,],
                                     [train_label, train_label, train_label, train_label, train_label,])
        # ** training loss is not stored ** #
        
    # epoch-end validation
    y_pred = model.predict([valid_input])
    record_temp = np.mean(keras.losses.categorical_crossentropy(valid_label, y_pred))
    # ** validation loss is not stored ** #
    
    # if loss is reduced
    if record - record_temp > min_del:
        print('Validation performance is improved from {} to {}'.format(record, record_temp))
        record = record_temp; # update the loss record
        tol = 0; # refresh early stopping patience
        # ** model checkpoint is not stored ** #
        
    # if loss not reduced
    else:
        print('Validation performance {} is NOT improved'.format(record_temp))
        tol += 1
        if tol >= max_tol:
            print('Early stopping')
            break;
        else:
            # Pass to the next epoch
            continue; 


# ## Evaluation
# 
# The testing set performance is evaluated with cross-entropy and example outputs.

# In[ ]:


path_test_img = "frames/testsegmentation/Images/"
path_test_mask = "frames/testsegmentation/Masks/"


# In[ ]:


test_input_names = np.array(sorted(glob(path_test_img+'*.jpg')))
test_label_names = np.array(sorted(glob(path_test_mask+'*.png')))


# In[ ]:


test_input = input_data_process(utils.image_to_array(test_input_names, size=512, channel=3))
test_label = target_data_process(utils.image_to_array(test_label_names, size=512, channel=1))


# In[ ]:


y_pred = model.predict([test_input])


# In[ ]:


print('Testing set cross-entropy = {}'.format(np.mean(keras.losses.categorical_crossentropy(test_label, y_pred))))


# **Example of outputs**
# 
# As a common practice in computer vision projects, only nice looking samples are plotted : |

# In[ ]:


i_sample = 3

fig, AX = plt.subplots(1, 3, figsize=(13, (13-0.2)/3))
plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
for ax in AX:
    ax = ax_decorate_box(ax)
AX[0].pcolormesh(np.mean(test_input[i_sample, ...,], axis=-1), cmap=plt.cm.gray)
AX[1].pcolormesh(y_pred[i_sample, ...,0], cmap=plt.cm.binary)
AX[2].pcolormesh(test_label[i_sample, ..., 0], cmap=plt.cm.binary_r)

AX[0].set_title("Original", fontsize=14);
AX[1].set_title("prediction", fontsize=14);
AX[2].set_title("Labeled truth", fontsize=14);


# ## Discussion
# 
# A segmentation model is proposed based on the architecture of UNET 3+ and is trained using the Oxford-IIIT Pets dataset. Result evaluation indicates that this segmentation model can distinguish pixes of a pet from image backgrounds.
# 
# Many technical details of this work, for example, network hyper-parameters and training strategy, can be improved for achieving better performance. 

# In[ ]:


model.summary()


# **Visualizing model performance**

# precision recall

# ### **Let's evaluate the model on our test data!**

# In[ ]:


y_pred = model.predict([test_input])


# In[ ]:


model.evaluate(test_input, test_label, return_dict= True)


# y_true	The ground truth values, with the same dimensions as y_pred. Will be cast to bool.
# y_pred	The predicted values. Each element must be in the range [0, 1].
# sample_weight	Optional weighting of each example. Defaults to 1. Can be a Tensor whose rank is either 0, or the same rank as y_true, and must be broadcastable to y_true.

# In[ ]:



m = tf.keras.metrics.Recall()
m.update_state( test_label, y_pred, sample_weight=None)
m.result().numpy()


# In[ ]:


m = tf.keras.metrics.Precision()
m.update_state( test_label, y_pred, sample_weight=None)
m.result().numpy()


# In[ ]:


m = tf.keras.metrics.BinaryAccuracy()
m.update_state( test_label, y_pred, sample_weight=None)
m.result().numpy()

