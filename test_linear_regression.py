import tensorflow as tf
# Tensorflow basics:
# https://www.datacamp.com/community/tutorials/tensorflow-tutorial
# https://medium.com/tensorflow/mit-deep-learning-basics-introduction-and-overview-with-tensorflow-355bcd26baf0
# https://github.com/lexfridman/mit-deep-learning

# You can create constants in TF to hold specific values
a = tf.constant(1)
b = tf.constant(2)
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
# Adding and multiplying the constants
c = a + b
d = a * b

# Everything so far has just specified the TensorFlow graph. We haven’t yet computed anything. To do this, we need to start a session in which the computations will take place. The following code creates a new session
sess = tf.Session()

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)


x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})