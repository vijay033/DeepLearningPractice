
# coding: utf-8

# In[1]:


#importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import regularizers


# In[2]:


vgg16_model = keras.applications.vgg16.VGG16()


# In[3]:


vgg16_model.summary()


# In[4]:


type(vgg16_model)


# In[5]:


vgg16_model.layers.pop()
vgg16_model.summary()


# In[6]:


model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


# In[7]:


model.summary()


# In[8]:


for layer in model.layers:
    layer.trainable = False


# In[9]:


#model.add(Dense(10,activation= 'softmax',kernel_regularizer=regularizers.l2(0.01)))


# In[10]:


model.add(Dense(10,activation= 'softmax'))


# In[11]:


model.summary()


# In[12]:


# Compiling the CNN
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
#model.optimizer.lr = 0.001


# In[13]:


# Fitting the CNN to the images
# Image augmentation for reducing overfitting - Balancing bias and variance

from keras.preprocessing.image import ImageDataGenerator


# In[14]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[15]:


train_generator = train_datagen.flow_from_directory('Monkey/training',
                                                    target_size=(224, 224),
                                                    batch_size= 32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('Monkey/validation',
                                                        target_size=(224,224),
                                                        batch_size=16,
                                                        class_mode='categorical')


# In[17]:


#steps_per_epoch - no of training data image
#validation_steps - no of testing data image
model.fit_generator(train_generator,
                         steps_per_epoch=34,       #1097/32
                         epochs=100,
                         validation_data=validation_generator,
                         validation_steps=17)   #272/16


# In[61]:


import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from numpy import array
from numpy import argmax
from sklearn.metrics import roc_curve, roc_auc_score, auc


# In[70]:


test_imgs,test_labels = next(validation_generator)
predicted = model.predict_classes(test_imgs)
predicted_ = np.argmax(to_categorical(predicted), axis=1)
print(predicted_)
print(predicted_.shape)
#print(test_labels)

preditions = model.predict_proba(test_imgs)

rrow = len(test_labels)
ccol = np.size(test_labels,1) 

# print(rrow)
# print(ccol)

actual_labels = np.arange(rrow)
actual_labels.reshape(rrow,)
print(actual_labels.shape)



for i in range(rrow):
     for j in range(ccol):
         if test_labels[i][j] == 1:
             actual_labels[i] = j
            
for i, j in zip(predicted_,actual_labels):
    print( "Predicted class for monkey is {}, and the actual class of monkey is {}".format(i,j))

# # Calculate total roc auc score
# score = roc_auc_score(test_labels,predicted_.reshape(-1,1))
# print("Total roc auc score = {0:0.4f}".format(score))
fpr, tpr, _ = roc_curve(test_labels, predictions[:,1])
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual_labels,predicted_)
print(cm)

