#!/usr/bin/env python
# coding: utf-8

# <h1>Importing Tensorflow</h1>

# In[1]:


import tensorflow as tf


# <h1>Callback On The Training</h1>

# In[2]:


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# <h1>Callback object & Loading MNIST dataset</h1>

# In[3]:


callbacks = myCallback()
data = tf.keras.datasets.fashion_mnist


# <h1>Training set & Test set</h1>

# In[4]:


(training_images, training_labels), (test_images, test_labels) = data.load_data()


# <h1>Normalizing the Image </h1>

# In[5]:


training_images = training_images / 255.0
test_images = test_images / 255.0


# <h1>Defining Neural Network</h1>

# In[6]:


model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=(28, 28)),
 tf.keras.layers.Dense(128, activation=tf.nn.relu),
 tf.keras.layers.Dense(10, activation=tf.nn.softmax)
 ])


# <h1>Compiling Model</h1>

# In[7]:


model.compile(optimizer='adam',
     loss='sparse_categorical_crossentropy',
     metrics=['accuracy'])


# <h1>Training model by fitting the training images</h1>

# In[8]:


model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])


# <h1>Evaluting the model</h1>

# In[9]:


model.evaluate(test_images, test_labels)


# <h1>Exploring the Model Output</h1>

# In[10]:


classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

