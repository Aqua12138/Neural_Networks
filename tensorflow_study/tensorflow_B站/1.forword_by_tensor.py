import os
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽无关信息



class forword():
    def __init__(self):
        self.b = 128
        self.lr = 1e-3
        self.step = 0
        self.get_data()
        self.make_forword_net()
    #获取数据
    def get_data(self):
        # x:[60000,28,28]
        # y:[60000]
        (x, y), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
        #plt.imshow(x[0,:,:])
        #plt.show()
        #x:[0~255] => [0~1.]
        self.x = tf.convert_to_tensor(x,dtype=tf.float32) /255.
        self.y = tf.convert_to_tensor(y,dtype=tf.int32)
        self.x_test = tf.convert_to_tensor(x_test,dtype=tf.float32) /255.
        self.y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
        # print(x.shape,y.shape,x.dtype,y.dtype)
        # print(tf.reduce_min(x),tf.reduce_max(x))
        # print(tf.reduce_min(y),tf.reduce_max(y))
        #
        #
        self.train_db = tf.data.Dataset.from_tensor_slices((self.x,self.y)).batch(128) #创建一个dataset类型并分每批128
        self.test_db = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(128)
        # train_iter = iter(train_db)#实例化一个迭代器指向train_db的头
        # sample = next(train_iter)#迭代到下一个位置
        # print('batch:',sample[0].shape,sample[1].shape)
    def make_forword_net(self):
        #[b,784]=>[b,256]=>[b,128]=>[b,10]
        #Weight:[dim_in,dim_out],bias:[dim_out]
        w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
        b1 = tf.Variable(tf.zeros([256]),dtype=tf.float32)
        w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
        b2 = tf.Variable(tf.zeros([128]),dtype=tf.float32)
        w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
        b3 = tf.Variable(tf.zeros([10]),dtype=tf.float32)
        for epoch in range(10):#对整个数据集迭代
            for (self.x,self.y) in self.train_db:
                # x:[128,28,28]
                # y:[128,]
                # [b,28,28]=>[b,28*28]
                self.step += 1

                self.x = tf.reshape(self.x, [-1, 28 * 28])
                self.x = tf.cast(self.x,dtype=tf.float32)#self.train_db把数据变成u
                # x:[b,28*28]
                # h1 =x@w1 + b1
                # [b,28*28]@[784,256]+[256] =>[b,256]+[b,256] =>[b,256]
                with tf.GradientTape() as tape:#把所有徐亚进行梯度计算的都放到这个函数下
                    h1 = self.x@w1+b1
                    h1 = tf.nn.relu(h1)
                    # [b,256]=>[b,128]
                    h2 = h1@w2 +b2
                    h2 = tf.nn.relu(h2)
                    # [b,128]=>[b,10]
                    out = h2@w3 +b3

                    #compute loss
                    # out:[b,10]
                    # y:[b]
                    y_onehot = tf.one_hot(self.y,depth=10)
                    #mse = mean(sum(y-out)^2)
                    #[b,10]
                    loss = tf.square(y_onehot - out)#计算处每个样本的均方误差
                    # mean:scaler
                    loss = tf.reduce_mean(loss)/self.b/10#128个样本全部像素点的平均误差
                #使用损失函数loss来计算梯度
                grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
                # w1 = w1 - lr * w1_grad
                w1.assign_sub(self.lr * grads[0])#原地更新  w2 = w2 - self.lr * grads[0]这样赋值给了新的对象，Variable会变成tensor
                w2.assign_sub(self.lr * grads[2])
                w3.assign_sub(self.lr * grads[4])
                b1.assign_sub(self.lr * grads[1])
                b2.assign_sub(self.lr * grads[3])
                b3.assign_sub(self.lr * grads[5])

                if self.step % 100 == 0:
                    print(epoch,self.step,'loss',float(loss))
            total_correct ,total_num = 0,0
            for step,(x,y) in enumerate(self.test_db):#enumerate=>((0,number),(1,bumber)...)
                #[b,28,28]=>[b,28*28]
                x=tf.reshape(x,[-1,28*28])

                #[b,784]=>[b,256]=>[b,128]=>[b,10]
                h1=tf.nn.relu(x@w1+b1)
                h2=tf.nn.relu(h1@w2+b2)
                out = h2@w3 + b3

                # out:[b,10]~R
                # prob:[b,10]~[0,1]
                prob = tf.nn.softmax(out,axis=1)#把输出转换成概率
                #pred 返回了int64
                pred = tf.argmax(out,axis=1)#获得每个样本最大的概率的索引
                pred = tf.cast(pred,dtype=tf.int32)
                #y:[b]
                #[b],int32
                correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
                correct = tf.reduce_sum(correct)

                total_correct += int(correct)
                total_num += x.shape[0]

            acc = total_correct/total_num
            print('acc:',acc)

a = forword()