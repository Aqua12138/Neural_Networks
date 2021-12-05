import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

tf.random.set_seed(2345)
np.random.seed(2345)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽无关信息
#
batchsz = 128
#the most frequest world
total_words = 10000
#the longest of sequence
max_review_len = 80
#单词维度
embedding_len = 100
#hidden
unit=64
(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# x_train = tf.convert_to_tensor(x_train)
# y_train = tf.convert_to_tensor(y_train)
# x_test = tf.convert_to_tensor(x_test)
# y_test = tf.convert_to_tensor(y_test)
#x_train:[b,80]
#x_test:[b,80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len)#设置x的最大长度为80，长的部分截取，短的部分补零
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)

class MyRNN(keras.Model):
    # Cell方式构建多层网络
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # 词向量编码 [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        # 构建RNN
        self.rnn = keras.Sequential([
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True),
            layers.SimpleRNN(units, dropout=0.5)
        ])
        # 构建分类网络，用于将CELL的输出特征进行分类，2分类
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = keras.Sequential([
        	layers.Dense(32),
        	layers.Dropout(rate=0.5),
        	layers.ReLU(),
        	layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs # [b, 80]
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute,[b, 80, 100] => [b, 64]
        x = self.rnn(x)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.outlayer(x,training)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob
def main():
    units = 64
    epoch = 4

    model = MyRNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss = tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train,epochs=epoch,validation_data=db_test)

    model.evaluate(db_test)
if __name__ == '__main__':
    main()