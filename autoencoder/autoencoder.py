import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def gen_normal_sin():
   MAX_TIME_ERROR = 5
   MAX_VAL_ERROR = 0.01
   delay = int(MAX_TIME_ERROR * (np.random.rand(1)[0] - 0.5) / 0.5 + 0.5)
   data = []
   for i in range(200):
       data.append(np.sin(np.pi / 2 * (i - delay) / 40) + MAX_VAL_ERROR * (np.random.rand(1)[0] - 0.5) / 0.5)
   return data

def gen_abnormal_sin():
   MAX_TIME_ERROR = 20
   MAX_VAL_ERROR = 0.1
   delay = int(MAX_TIME_ERROR * (np.random.rand(1)[0] - 0.5) / 0.5 + 0.5)
   data = []
   for i in range(200):
       data.append(np.sin(np.pi / 2 * (i - delay) / 40) + MAX_VAL_ERROR * (np.random.rand(1)[0] - 0.5) / 0.5)
   return data

def plot_data(x, y, pos):
   plt.subplot(pos)
   plt.plot(x, y)
   plt.xlabel('Time')
   plt.ylabel('sin(x)')
   plt.axis('tight')

#Generate a dataset
np.random.seed(0)

x = []
for i in range(1000):
   x.append(gen_normal_sin())
       
num_input_nodes = len(x[0])

#we can tune these 3 parameters
num_hidden_nodes = 64
learning_rate = 0.01
epochs = 2000

nn_input = tf.placeholder(tf.float32,
                       shape=[None, num_input_nodes],
                       name="input")

#Define weights and biases of the neural network
weights = {
   'hidden': tf.Variable(tf.random_normal([num_input_nodes, num_hidden_nodes], seed=0)),
   'output': tf.Variable(tf.random_normal([num_hidden_nodes, num_input_nodes], seed=0))
}

biases = {
   'hidden': tf.Variable(tf.random_normal([num_hidden_nodes], seed=0)),
   'output': tf.Variable(tf.random_normal([num_input_nodes], seed=0))
}

#define activation function in hidden layer
hidden_layer = tf.sigmoid(tf.add(tf.matmul(nn_input, weights['hidden']), biases['hidden']))
#define activation function in output layer
output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(nn_input - output_layer, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
   # create initialized variables
   sess.run(tf.global_variables_initializer())
   for epoch in range(epochs):
       c = sess.run([optimizer, cost], feed_dict = {nn_input: x})
   
   data = gen_normal_sin()
   result = sess.run(output_layer, feed_dict = {nn_input: [data]})
   x = np.linspace(0, 4, 200)
   plot_data(x, data, 221)
   plot_data(x, result[0], 222)
   
   wrong_data = gen_abnormal_sin()
   result2 = sess.run(output_layer, feed_dict = {nn_input: [wrong_data]})
   plot_data(x, wrong_data, 223)
   plot_data(x, result2[0], 224)
   plt.show()