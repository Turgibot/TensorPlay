"""
demonstrating the difference between lazy and normal variables loading
Created by Guy Tordjman (turgibot@gmail.com)
March 22nd 2018
Please always use your code for the benefit of humanity.
"""
import tensorflow as tf


########################################
# NORMAL LOADING
# print out a graph with 1 Add node
########################################

x = tf.Variable(100, name='x')
y = tf.Variable(200, name='y')
z = tf.add(x, y)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graphs/normal_loading', sess.graph)
    for _ in range(10):
        sess.run(z)
    print(tf.get_default_graph().as_graph_def())
    writer.close()

########################################
# LAZY LOADING
# print out a graph with 10 Add nodes
########################################

x = tf.Variable(100, name='x')
y = tf.Variable(200, name='y')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
    for _ in range(10):
        sess.run(tf.add(x, y))
    print(tf.get_default_graph().as_graph_def())
    writer.close()
