import tensorflow as tf
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽无关信息
batchsz = 128



(train_x,train_y),(test_x,test_y) = tf.keras.datasets.fashion_mnist.load_data()
print(train_x.shape)

#数据预处理函数
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

db = tf.data.Dataset.from_tensor_slices((train_x,train_y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)#传入数据预处理进行预处理,.batch是dataset中的一种分类方法，.shuffle表示打乱，10000是打乱的随机数

db_test = tf.data.Dataset.from_tensor_slices((test_x,test_y))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
print('batch',sample[0].shape,sample[1].shape)

#构建网络
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=256,activation=tf.nn.relu),#[b,784]=>[b,256]
    tf.keras.layers.Dense(units=128,activation=tf.nn.relu),#[b,256]=>[b,128]
    tf.keras.layers.Dense(units=64,activation=tf.nn.relu),#[b,128]=>[b,64]
    tf.keras.layers.Dense(units=32,activation=tf.nn.relu),#[b,64]=>[b,32]
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax),#[b,32]=>[b,10]
])

model.build(input_shape=[None,28*28])

optimizer = tf.keras.optimizers.Adam(lr=1e-3)#使用计算出来的梯度更新权值

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer("logs/")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)
def main():



        for epoch in range(30):

            for step,(x,y) in enumerate(db):
                #x:[b,28,28] = [b,784]
                #y:[b]
                x = tf.reshape(x,[-1,28*28])

                with tf.GradientTape() as tape:
                    #[b,784] => [b,10]

                    logits = model(x)
                    y_onehot = tf.one_hot(y,depth=10)
                    #[b]
                    loss = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
                    loss2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))

                grads = tape.gradient(loss2,model.trainable_variables)
                optimizer.apply_gradients(zip(grads,model.trainable_variables))#w = w-lr*grads,zip会把前后对应元素打包成元组
                #可视化




                if step % 100 == 0:
                    print(epoch,step,'loss:',loss,loss2)
                    with writer.as_default():
                        tf.summary.scalar("loss", float(loss), step=step)
                        tf.summary.scalar("loss2", float(loss), step=step)
            totle_correct = 0
            totle = 0
            for step,(x,y) in enumerate(db_test):
                x = tf.reshape(x,[-1,28*28])
                losgits = model(x)
                #[b,10]=>[b] int64
                pred = tf.argmax(losgits,axis=1)
                pred = tf.cast(pred,dtype=tf.int32)

                #pred:[b]
                #y:[b]
                # correct:[b]:True:equa;,False:not equa
                correct = tf.equal(pred,y)
                correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))

                totle_correct += int(correct)
                totle += x.shape[0]
            acc = totle_correct/totle
            print("acc:",acc)
            # with writer.as_default():
            #     tf.summary.scalar("loss", float(acc), step=epoch)

main()