import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')
n_nodes_h1=500
n_nodes_h2=500
n_classes=10
batch=100
def neural_network(data):
    hiddenl_1={'weights':tf.Variable(tf.random_normal([784,500])),
                'biases':tf.Variable(tf.random_normal([n_nodes_h1]))}
    hiddenl_2={'weights':tf.Variable(tf.random_normal([n_nodes_h1,500])),
                'biases':tf.Variable(tf.random_normal([n_nodes_h2]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_h2,n_classes])),
                  'biases':tf.Variable(tf.random_normal([n_classes]))}
    l1=tf.add(tf.matmul(data,hiddenl_1['weights']),hiddenl_1['biases'])
    l1=tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1,hiddenl_2['weights']),hiddenl_2['biases'])
    l2=tf.nn.relu(l2)
    output=tf.add(tf.matmul(l2,output_layer['weights']),output_layer['biases'])
    return output

def predicto(x):
    prediction=neural_network(x)
    
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimize=tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        hm_epo=10
        for i in range(hm_epo):
          loss=0
          for m in range(int(mnist.train.num_examples/batch)):
            epoch_x,epoch_y=mnist.train.next_batch(batch)

            o,c=sess.run([optimize,cost], feed_dict={x:epoch_x,y:epoch_y})
            print ("Cost:",c, "optimize",o)
            loss+=c

predicto(x)
