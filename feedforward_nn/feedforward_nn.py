import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Generate a dataset
np.random.seed(0)
x, y = sklearn.datasets.make_moons(200, noise=0.20)
#normalise test data so tensorflow is able to use
x_norm = np.float32(x).tolist()
y_norm = []
for i in range(len(y)):
    if y[i] == 0.0:
        y_norm.append([1.0, 0.0])
    else:
        y_norm.append([0.0, 1.0]) 
        
num_input_nodes = len(x[0])
num_output_nodes = 2

#we can tune these 3 parameters
num_hidden_nodes = 20
learning_rate = 0.01
epochs = 2000

nn_input = tf.placeholder(tf.float32,
                        shape=[None, num_input_nodes],
                        name="input")
target = tf.placeholder(tf.float32,
                        shape=[None, num_output_nodes],
                        name="output")

#Define weights and biases of the neural network
weights = {
    'hidden': tf.Variable(tf.random_normal([num_input_nodes, num_hidden_nodes], seed=0)),
    'output': tf.Variable(tf.random_normal([num_hidden_nodes, num_output_nodes], seed=0))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([num_hidden_nodes], seed=0)),
    'output': tf.Variable(tf.random_normal([num_output_nodes], seed=0))
}

#define activation function in hidden layer
hidden_layer = tf.sigmoid(tf.add(tf.matmul(nn_input, weights['hidden']), biases['hidden']))
#define activation function in output layer
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
#define cost of our neural network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output_layer))
#set the optimizer, i.e. our backpropogation algorithm. Here we use Adam, which is an efficient variant of Gradient Descent algorithm.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    # create initialized variables
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        c = sess.run([optimizer, cost], feed_dict = {nn_input: x_norm, target: y_norm}) 
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_norm, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "accuracy=", sess.run(accuracy, feed_dict={nn_input: x_norm, target: y_norm})
    
    #plotting boundary
    
    prediction = tf.argmax(output_layer, 1)
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.float32(np.c_[xx.ravel(), yy.ravel()]).tolist()
    z = np.array(sess.run(prediction, feed_dict={nn_input: grid}))
    z = z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)