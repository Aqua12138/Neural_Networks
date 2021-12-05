import tensorflow as tf

x = tf.random.normal([2,4])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(2)
])
model.build(input_shape=[None,4])#创建网络，生成w和b
model.summary()
for p in model.trainable_variables:
    print(p.shape,p.shape)
