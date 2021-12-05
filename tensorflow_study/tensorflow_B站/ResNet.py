import tensorflow as tf
class BasicBlock(tf.keras.layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')#stride 是采样的间隔，1就是全采样，2就是shape/2，padding是是否补全,如果shape是2，则补全到一半尺寸
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride !=1:
            #self.downsample = tf.keras.models.Sequential()
            self.downsample=tf.keras.layers.Conv2D(filter_num,(1,1),strides=stride)##
        else:
            self.downsample = lambda x:x

    def call(self,input,training = None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)

        output = tf.keras.layers.add([out,identity])#两个输出的网络层相加
        output = self.relu(output)#最后relu，因为relu没有自己的w、b，所以可以重复使用

        return output

class ResNet(tf.keras.Model):
    def __init__(self,layer_dims,num_classes=10):#[2,2,2,2]
        super(ResNet, self).__init__()
        self.stem = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3),strides = (1,1)),
                                                tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.Activation('relu'),
                                                tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride = 2)
        self.layer3 = self.build_resblock(256, layer_dims[2],stride = 2)
        self.layer4 = self.build_resblock(512, layer_dims[3],stride = 2)

        #output:[b,h,w,512]
        self.avgpool = tf.keras.layers.GlobalMaxPool2D()#用在不确定输出大小,会把所有的像素都均值化，【b,512,h,w】=>[b,512]
        self.fc = tf.keras.layers.Dense(num_classes)



    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #[b,c]
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self,filter_num,blocks,stride=1):
        res_blocks = tf.keras.models.Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num,stride))

        for _ in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num,stride=1))
        return res_blocks

def resnet18():
    return ResNet([2,2,2,2])
def resnet34():
    return ResNet([3,2,2,2])



