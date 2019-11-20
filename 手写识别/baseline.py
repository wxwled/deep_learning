# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:38:39 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:07:59 2019

@author: Administrator
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#原始数据读入
train = pd.read_csv('./data/train.csv')
test_x = pd.read_csv('./data/test.csv')
train_y = train['label']
del train['label']
train_x = train

#展示灰度矩阵图像
def show(gray_matrix):
    plt.subplot(111)
    plt.imshow(gray_matrix,cmap=plt.get_cmap('gray'))
    
def show_with_index(train_x,index):
    show(np.array(train_x.iloc[index]).reshape((28,28)))


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

#训练数据生成
X_train = np.array(train_x).reshape(-1,28,28,1)/255.0
X_test = np.array(test_x).reshape(-1,28,28,1)/255.0
y_train = np_utils.to_categorical(train_y, num_classes=10)

#模型设计(LeNet)
#28*28*1 -> 28*28*32(24M) -> 14*14*32(6M) -> 14*14*64(12M) -> 7*7*64(3M) -> 1024 -> 10
model = Sequential()

# Conv1 28*28*32
model.add(Convolution2D(32,5,strides=1,padding='same',batch_input_shape = (None, 28, 28, 1)))
model.add(Activation('relu'))
# Conv2 28*28*32
model.add(Convolution2D(32,5,strides=1,padding='same'))
model.add(Activation('relu'))
# pool 14*14*32
model.add(MaxPooling2D(2,2,'same'))
model.add(Dropout(0.25))

# Conv3 64
model.add(Convolution2D(64,3,strides=1,padding='same'))
model.add(Activation('relu'))
# Conv4 64
model.add(Convolution2D(64,3,strides=1,padding='same'))
model.add(Activation('relu'))
# pool 7*7*64
model.add(MaxPooling2D(2,2,'same'))
model.add(Dropout(0.25))
#4*4*256=4096
#数据平展成一维
model.add(Flatten())

#fc*2
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

#optimizer
adam = Adam(lr=0.0001)

#设置模型损失和优化器以及参考参数
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

print('Training ------------')
model.fit_generator(datagen.flow(X_train,y_train, batch_size=64), epochs=3,
                    steps_per_epoch=X_train.shape[0], callbacks=[learning_rate_reduction])


y_test=model.predict_classes(X_test)
submission = pd.read_csv('./data/sample_submission.csv')
submission['Label']=y_test
submission.to_csv('./data/submission.csv',index=0)
