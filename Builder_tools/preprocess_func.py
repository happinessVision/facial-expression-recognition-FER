#!/usr/bin/env python
# coding: utf-8

# In[26]:


import keras
import keras.backend as K
import numpy as np


# In[25]:


# Функция препроцессинга для изображений
# https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/utils.py 
# Для архитектуры ResNEt50
def preprocess_input_ResNet50(x, data_format=None):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        x_temp = x_temp[:, ::-1, ...]
        x_temp[:, 0, :, :] = x_temp[:, 0, :, :] - 91.4953
        x_temp[:, 1, :, :] = x_temp[:, 1, :, :] - 103.8827
        x_temp[:, 2, :, :] = x_temp[:, 2, :, :] - 131.0912
    else:
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] = x_temp[..., 0] - 91.4953
        x_temp[..., 1] = x_temp[..., 1] - 103.8827
        x_temp[..., 2] = x_temp[..., 2] - 131.0912
    
    return x_temp


# In[24]:


# Функция препроцессинга для изображений
# https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/utils.py 
# Для архитектуры VGG16
def preprocess_input_VGG16(x, data_format=None):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        x_temp = x_temp[:, ::-1, ...]
        x_temp[:, 0, :, :] = x_temp[:, 0, :, :] - 93.5940
        x_temp[:, 1, :, :] = x_temp[:, 1, :, :] - 104.7624
        x_temp[:, 2, :, :] = x_temp[:, 2, :, :] - 129.1863
    else:
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] = x_temp[..., 0] - 93.5940
        x_temp[..., 1] = x_temp[..., 1] - 104.7624
        x_temp[..., 2] = x_temp[..., 2] - 129.1863

    return x_temp

