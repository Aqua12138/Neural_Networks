import tensorflow as tf
import os
import ssl
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'https://127.0.0.1:7890'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽无关信息
class Cifar:
    def __init__(self):
        self.batchsz = 128
        self.get_data()
        pass

    def preprecess(self,x,y):
        #属于预处理
        x = tf.cast(x,dtype=tf.float32)/255.-0.5
        y = tf.cast(y,dtype=tf.int32)
        return x,y

    def get_data(self):
        (x,y),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
        y = tf.squeeze(y)#修剪维度
        y_test = tf.squeeze(y)
        y = tf.one_hot(y,depth=10)
        y_test = tf.one_hot(y_test,depth=10)
        print('datasets',x.shape,y.shape,x.min(),x.max())

        train_db = tf.data.Dataset.from_tensor_slices((x,y))
        self.train_db = train_db.map(self.preprecess).shuffle(10000).batch(batch_size=self.batchsz)
        test_db = tf.data.Dataset.from_tensor_slices((x,y))
        self.test_db = test_db.map(self.preprecess).batch(batch_size=self.batchsz)

class MyDense(tf.keras.layers.Layer):
    def __init__(self,inp_dim,outp_dim):
        #自定义层
        super(MyDense,self).__init__()
        self.kernel = self.add_weight('w',[inp_dim,outp_dim])
        # self.bias = self.add_variable('b',[outp_dim])#不加bias
    def call(self,inputs,training=None):
        x = inputs @ self.kernel
        return x
class Mynetwork(tf.keras.Model):
    #自定义网络
    def __init__(self):
        #实例化自定义层
        super(Mynetwork,self).__init__()
        self.fc1 = MyDense(32*32*3,512)
        self.fc2 = MyDense(512,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)
    def call(self,inputs,training=None):
        #[b,32,32,3]
        x = tf.reshape(inputs,[-1,32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x
cifar = Cifar()
network = Mynetwork()
network.compile(optimizer=tf.optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/4.cifar10_weight.ckpt')
network.fit(cifar.train_db,epochs=5,validation_data=cifar.test_db,validation_freq=1)

network.evaluate(cifar.test_db)
network.save_weights('ckpt/4.cifar10_weight.ckpt')
del network
print('save to ckpt')
#保存
network = Mynetwork()
network.compile(optimizer=tf.optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/4.cifar10_weight.ckpt')
network.evaluate(cifar.test_db)