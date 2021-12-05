import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

tf.random.set_seed(2345)
np.random.seed(2345)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽无关信息
#
batchsz = 128
# the most frequest world
total_words = 10000
# the longest of sequence
max_review_len = 80
# 单词维度
embedding_len = 100
# hidden
unit = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# x_train = tf.convert_to_tensor(x_train)
# y_train = tf.convert_to_tensor(y_train)
# x_test = tf.convert_to_tensor(x_test)
# y_test = tf.convert_to_tensor(y_test)
# x_train:[b,80]
# x_test:[b,80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)  # 设置x的最大长度为80，长的部分截取，短的部分补零
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        self.state0 = [tf.zeros([batchsz, units]), tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units]), tf.zeros([batchsz, units])]
        # transform text to embedding representation[b,80]=>[b,80,100]
        self.embeding = tf.keras.layers.Embedding(total_words, embedding_len,
                                                  input_length=max_review_len)  # (所有词数（样本种类），每个样本的特征维度，最大语句长度（考虑之前最长的样本长度）)
        # cell [b,80,100],h_dim:64
        self.cell1 = layers.LSTMCell(units, dropout=0.5)
        self.cell2 = layers.LSTMCell(units, dropout=0.5)

        # fc,[b,80,100]=>[b,64]=>[b,1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # inputs[b,80]
        '''
        net(x) net(x,traning=True):train node
        net(x,training=False):test
        :param inputs: [b,80]
        :param training:
        :param mask:
        :return:
        '''
        # [b,80]
        x = inputs
        # embedding:[b,80]=>[b,80,100]
        x = self.embeding(x)
        # rnn cell compute
        # [b,80,100]=>[b,64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word:[b,100] 提取第二维度 循环80次
            # h1 = x*wxh+h*whh [b,100]=>[b,64]
            out, state0 = self.cell1(word, state0, training)
            # [b,64]=>[b,64]
            out1, state1 = self.cell2(out, state1)
        # out:[b,64]
        x = self.outlayer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)
        return prob


def main():
    units = 64
    epoch = 4

    import time
    t0 = time.time()
    model = MyRNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epoch, validation_data=db_test)

    model.evaluate(db_test)
    t1 = time.time()
    print('total time cost:', t1 - t0)


if __name__ == '__main__':
    main()