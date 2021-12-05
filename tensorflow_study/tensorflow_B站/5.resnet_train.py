import ResNet as rs
import tensorflow as tf
import os

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

cifar = Cifar()
model = rs.resnet18()
model.build(input_shape=(None,32,32,3))
optimizer = tf.keras.optimizers.Adam(lr=1e-3)

for epoch in range(50):
    for step,(x,y) in enumerate(cifar.train_db):
        with tf.GradientTape() as tape:
            #[b,32,32,3]=>[b,10]
            logits = model(x)
            loss = tf.losses.categorical_crossentropy(y,logits,from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step%100 == 0:
            print(epoch,step,'loss',float(loss))

    totle_correct = 0
    totle = 0
    for step, (x, y) in enumerate(cifar.db_test):
        x = tf.reshape(x, [-1, 28 * 28])
        losgits = model(x)
        # [b,10]=>[b] int64
        pred = tf.argmax(losgits, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        # pred:[b]
        # y:[b]
        # correct:[b]:True:equa;,False:not equa
        correct = tf.equal(pred, y)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

        totle_correct += int(correct)
        totle += x.shape[0]
    acc = totle_correct / totle
    print("acc:", acc)
