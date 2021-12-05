import os
import tensorflow as tf
import numpy as np

#Instantiate the data
fashion = tf.keras.datasets.fashion_mnist

#down load the data
(train_images,train_labels),(test_images,test_labels) = fashion.load_data()

#resize the images data from 3 dimension to 4 dimension
train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))


if(os.path.exists('/home/zhx/tensorflow_model/model2')):
    model = tf.keras.models.load_model('/home/zhx/tensorflow_model/model2')

else:
    #build the model
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                 tf.keras.layers.MaxPool2D(2,2),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128,activation='relu'),
                                 tf.keras.layers.Dense(10,activation='relu')])

    model.summary()

    #define the compile rule
    model.compile(optimizer='Adam',loss = 'sparse_categorical_crossentropy')


#start to train
history = model.fit(train_images,train_labels,epochs=20)

model.save('～/tensorflow_model/model2',save_format='tf')