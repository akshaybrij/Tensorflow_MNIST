import numpy as np
import tensorflow as tf
#from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
learning_rate=0.001;
batch_size=128;
display_steps=10;
training_iter_size=20000;
n_input=784
n_classes=10
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(X,w,b):
    x=tf.reshape(X,shape=[-1,28,28,1])
    conv1=conv2d(x,w['w1'],b['b1'])
    conv1=max_pool2d(conv1)
    conv2=conv2d(conv1,w['w2'],b['b2'])
    conv2=max_pool2d(conv2)
    fc1=tf.reshape(conv2,[-1,w['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,w['wd1']),b['bd1'])
    fc1=tf.nn.relu(fc1)
    outs=tf.add(tf.matmul(fc1,w['out']),b['out'])
    return outs

w={
'w1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
"w2":tf.Variable(tf.random_normal([5,5,32,64])),
"wd1":tf.Variable(tf.random_normal([7*7*64,1024])),
"out":tf.Variable(tf.random_normal([1024,n_classes]))
}
b={
"b1":tf.Variable(tf.random_normal([32])),
"b2":tf.Variable(tf.random_normal([64])),
"bd1":tf.Variable(tf.random_normal([1024])),
"out":tf.Variable(tf.random_normal([n_classes])),
}
pred = conv_net(x,w,b)
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
c_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(c_pred,tf.float32))

init=tf.initialize_all_variables();
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size <= training_iter_size:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%10 == 0:
            loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y})
            print "Loss {} Accuracy {}".format(loss,acc)
        step+=1
