import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential#model tipi 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout#Katmanlar
from tensorflow.keras.optimizers import Adam #İmage için iyi optimizer
from tensorflow.keras.losses import categorical_crossentropy #Loss hesabı
from tensorflow.keras.utils import to_categorical, plot_model #Kategorileri ayarlıyor (onehotencoder gibi)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pydicom as pdc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
#MENINGIOMA = 1 

data_path = 'C:\\Users\osman\Desktop\YSA_Proje\\'

dim = (256,256)


data = np.loadtxt(data_path + "data256.csv") 
data = data.reshape(data.shape[0], data.shape[1] // dim[0], dim[0])


labels = np.loadtxt(data_path +'labels.csv')

x_train, x_test, y_train, y_test = train_test_split(data, labels , test_size=0.2, random_state=42)

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# x_valid = x_train[:int(x_train.shape[0]*validrate)]
# y_valid = y_train[:int(y_train.shape[0]*validrate)]

# x_train = x_train[int(x_train.shape[0]*validrate):]
# y_train = y_train[int(y_train.shape[0]*validrate):]

x_train = x_train.reshape(x_train.shape[0],256,256,1)
# x_valid = x_valid.reshape(x_valid.shape[0],256,256,1)
x_test  = x_test.reshape(x_test.shape[0],256,256,1)

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 3) 
y_test = to_categorical(y_test, 3)

a = x_train[5]
np.amax(a)
# x = 1
# plt.imshow(x_train[x])

#--------------------------------------DATA HAZIRLIĞI SONU---------------------

import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 42} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

model = Sequential()

model.add((Conv2D(filters = 16 , kernel_size =(3,3), activation = 'relu', input_shape = (256,256,1))))
model.add((Conv2D(filters = 32 , kernel_size =(3,3), activation = 'relu')))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64 , kernel_size =(3,3),  activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 128 , kernel_size =(3,3),  activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 128 , kernel_size =(3,3),  activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.summary()

model.compile(  loss = "categorical_crossentropy",
                optimizer = Adam(learning_rate = 0.001),
                metrics = ['accuracy'], 
                )

earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
checkpointer = ModelCheckpoint(filepath="bestmodel.h5", verbose=1, save_best_only=True)

#----------------------------------MODEL------------------------------------

history = model.fit(x_train,y_train,                    
                    batch_size = 32,#Datasetten tek seferde kaç resim alacak??Ram ile alakalı               
                    epochs = 22,                    
                    verbose = '1',
                    callbacks = [checkpointer, earlystopping],
                    validation_data = (x_test,y_test)
                    )
score = model.evaluate(x_test, y_test, verbose = 1)

print("Test Loss: ", score[0])
print('Test Accuracy: ', score[1])

print(history.history.keys())

#Accuracy için 
plt.figure()
plt.plot(history.history['acc'][:19])
plt.plot(history.history['val_acc'][:19])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc = 'lower right')
plt.show
#Loss için
plt.figure()
plt.plot(history.history['loss'][:19])
plt.plot(history.history['val_loss'][:19])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc = 'upper right')
plt.show

print(history.history)



"""
#---------------------------------Test--------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import Preprocessing as pp
import cv2 as cv

t =0
import mat73
# test1= mat73.loadmat(data_path +'data\{}.mat'.format(t))
test1 = x_test[t]
# test1 = img_to_array(test1) / 255
# plt.imshow(test1)

test1 = np.expand_dims(test1, axis = 0)

pred = model.predict(test1)
print("Tahmin edilen class: ",+ pred.argmax())
print("Gerçek class: ",+ y_test[t])

# x66 = 0.70434
# x66loss =0.667283

# xpp =0.7304
# xpploss =0.65113

# x256 = 0.93043
# x256loss =0.2210
"""


