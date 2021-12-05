import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
import pydot
class build_model():
    #use sequential build model
    def build_model_with_sequential(self):
        #instantiate a Sequential
        seq_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                                tf.keras.layers.Dense(128,activation=tf.nn.relu),
                                                tf.keras.layers.Dense(128,activation=tf.nn.softmax)])
        return seq_model


    #using function to build model
    def build_model_with_function(self):
        #instantiate a input Tensor
        input_layer = tf.keras.Input(shape=(28,28))
        first_layer = tf.keras.layers.Flatten()(input_layer)
        first_dense = tf.keras.layers.Dense(128,activation=tf.nn.relu)(first_layer)
        output_layer = tf.keras.layers.Dense(10,activation=tf.nn.softmax)(first_dense)

        #declare inputs and outputs
        func_model = Model(inputs = input_layer,outputs = output_layer)

        return func_model


M = build_model()
function_model = M.build_model_with_function()

plot_model(function_model,show_shapes=True,show_layer_names=True,to_file='/home/zhx/tensorflow_model/model_picture5.png')

mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
train_images = train_images/255.
test_images = test_images/255.
function_model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
function_model.fit(train_images,train_labels,epochs=5)

#evaluate the model
function_model.evaluate(test_images,test_labels)
