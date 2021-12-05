import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Model():
    def get_data(self):
        dir_path = '/home/zhx/下载/ENB2012_data.xlsx'
        data = pd.read_excel(dir_path)
        data_x = data.iloc[:,:8]
        data_y = data.iloc[:,8:]

        #split the data for training and testing

        x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=2)
        #insistiate the StandSclar
        est = StandardScaler()
        x_train = est.fit_transform(x_train)
        x_test = est.fit_transform(x_test)
        return x_train,x_test,y_train,y_test
    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(8,))
        first_dense = tf.keras.layers.Dense(128,activation='relu')(input_layer)
        second_dense = tf.keras.layers.Dense(128,activation='relu')(first_dense)

        #define multiple out_put layer
        y1_output = tf.keras.layers.Dense(1,name='y1_output')(second_dense)
        third_dense = tf.keras.layers.Dense(64,activation='relu')(second_dense)

        y2_output = tf.keras.layers.Dense(1,name='y2_output')(third_dense)

        #declare the output and input
        model = tf.keras.models.Model(inputs=input_layer,outputs=[y1_output,y2_output])
        model.summary()
        return model
    #define the compile rule
    def model_compile(self,model):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
        model.compile(optimizer = optimizer,
                      loss = {'y1_output':'mse','y2_output':'mse'},
                      metrics = {'y1_output':tf.keras.metrics.RootMeanSquaredError(),
                                 'y2_output':tf.keras.metrics.RootMeanSquaredError()})

M = Model()
x_train,x_test,y_train,y_test = M.get_data()
model = M.build_model()
M.model_compile(model)
history = model.fit(x_train,y_train,epochs=500,batch_size=1,validation_data=(x_test,y_test))#tch_size is how may time to train all data

