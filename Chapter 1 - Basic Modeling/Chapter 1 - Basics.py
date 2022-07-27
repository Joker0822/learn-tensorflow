#!/usr/bin/env python
# coding: utf-8

# <h1>Installing TensorFlow</h1>

# In[1]:


get_ipython().system('pip install tensorflow')


# <h1>Importing the Libraries</h1>

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# <h1>Specifying Layer, Loss Function and Optimizer</h1>

# In[3]:


l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')


# <h1>Numpy Arrays</h1>

# In[4]:


xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# <h1>Learning Process</h1>

# In[5]:


model.fit(xs, ys, epochs=500)


# <h1>Prediction & what model learnt</h1>

# In[6]:


print(model.predict([10.0]))
print("Here is what I learned: {}".format(l0.get_weights()))

