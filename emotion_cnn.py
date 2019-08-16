import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers
class Emotion_cnn():
    def __init__(self):
        with tf.name_scope("parameters"):
            self.class_num = 7
            self.channel = 1
            self.hidden_dim = 1024
            self.full_shape = 2304
            self.optimizer_ = 'Adam'
        with tf.name_scope("input"):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channel], name='x_input')
        with tf.name_scope("y_target"):
            self.y_target = tf.placeholder(dtype=tf.int32, shape=[None], name='y_target')
        with tf.name_scope("cnn_structure"):
            self.pool_size = [0, 3, 3, 3]
            self.pool_stride = [0, 2, 2, 2]
            self.maxpool_ksize = [1, 3, 3, 1]
            self.maxpool_strides = [1, 2, 2, 1]
            self.padding = "SAME"
            self.activation = "relu"
            self.initializer = initializers.xavier_initializer()
            self.filter = [[1, 1, 1, 32], [5, 5, 32, 32], [3, 3, 32, 32], [5, 5, 32, 64]]
            self.bias = [32, 32, 32, 64]
            self.strides = [1, 1, 1, 1]
            self.batch_size = tf.shape(self.x_input)[0]
        with tf.name_scope("hyperparameters"):
            self.lr = 0.001
            self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
            self.depth_radius = 5
            self.norm_bias = 2.0
            self.alpha = 1e-3
            self.beta = 0.75
#         with tf.name_scope("evaluate"):
#         self.loss, self.accuracy = self.loss_accuracy(self.output_layer(self.cnn_layer()), self.y_target)
    def weight_variable(self, name, shape):
        with tf.variable_scope("weight"):
            weight = tf.get_variable(name, shape, dtype=tf.float32, initializer=self.initializer)
        return weight
    def bias_variable(self, name, shape):
        with tf.variable_scope("bias"):
            bias = tf.get_variable(name, shape, dtype=tf.float32, initializer=self.initializer)
        return bias
    def cnn_layer(self):
        with tf.name_scope("conv1"):
            conv1_weight = self.weight_variable('conv1_weight', self.filter[0])
            conv1_bias = self.bias_variable('conv1_bias', self.bias[0])
            conv1 = tf.nn.conv2d(self.x_input, conv1_weight, self.strides, self.padding)
            conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
            norm1 = tf.nn.lrn(conv1_relu, self.depth_radius, self.norm_bias, self.alpha, self.beta, 'norm1')
        with tf.name_scope("conv2"):
            conv2_weight = self.weight_variable('conv2_weight', self.filter[1])
            conv2_bias = self.bias_variable('conv2_bias', self.bias[1])
            conv2 = tf.nn.conv2d(norm1, conv2_weight, self.strides, self.padding)
            conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
            pool2 = tf.nn.max_pool(conv2_relu, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer2')
            norm2 = tf.nn.lrn(pool2, self.depth_radius, self.norm_bias, self.alpha, self.beta, 'norm2')
        with tf.name_scope("conv3"):
            conv3_weight = self.weight_variable('conv3_weight', self.filter[2])
            conv3_bias = self.bias_variable('conv3_bias', self.bias[2])
            conv3 = tf.nn.conv2d(norm2, conv3_weight, self.strides, self.padding)
            conv3_relu = tf.nn.relu(tf.nn.bias_add(conv3, conv3_bias))
            pool3 = tf.nn.max_pool(conv3_relu, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer3')
            norm3 = tf.nn.lrn(pool3, self.depth_radius, self.norm_bias, self.alpha, self.beta, 'norm3')
        with tf.name_scope("conv4"):
            conv4_weight = self.weight_variable('conv4_weight', self.filter[3])
            conv4_bias = self.bias_variable('conv4_bias', self.bias[3])
            conv4 = tf.nn.conv2d(norm3, conv4_weight, self.strides, self.padding)
            conv4_relu = tf.nn.relu(tf.nn.bias_add(conv4, conv4_bias))
            pool4 = tf.nn.max_pool(conv4_relu, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer4')
            norm4 = tf.nn.lrn(pool4, self.depth_radius, self.norm_bias, self.alpha, self.beta, 'norm4')
        return norm4
    def output_layer(self, fc_input):
        with tf.name_scope("output_layer"):
            with tf.name_scope("fully_connected_layer1"):
                fc_input = tf.reshape(fc_input, [self.batch_size, -1])
                fc1_weight = self.weight_variable('fc1_weight', [self.full_shape, self.hidden_dim * 2])
                fc1_bias = self.bias_variable('fc1_bias', [self.hidden_dim * 2])
                fc1 = tf.add(tf.matmul(fc_input, fc1_weight), fc1_bias)
            with tf.name_scope("fully_connected_layer2"):
                output1 = tf.nn.dropout(tf.nn.relu(fc1), keep_prob = self.dropout)
                fc2_weight = self.weight_variable('fc2_weight', [self.hidden_dim * 2, self.hidden_dim])
                fc2_bias = self.bias_variable('fc2_bias', [self.hidden_dim])
                fc2 = tf.add(tf.matmul(output1, fc2_weight), fc2_bias)
                output2 = tf.nn.dropout(tf.nn.relu(fc2), keep_prob = self.dropout)
            with tf.name_scope("softmax_layer"):
                out_weight = self.weight_variable('out_weight', [self.hidden_dim, self.class_num])
                out_bias = self.bias_variable('out_bias', [self.class_num])
                fc_output = tf.add(tf.matmul(output2, out_weight), out_bias, name = 'fc_output')
        return fc_output
    def loss_accuracy(self, logits_, label_):
        with tf.name_scope("loss"):
            loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_, labels = label_), name = 'cross_entropy_loss')
        with tf.name_scope("evaluate"):
            softmax = tf.nn.softmax(logits_, name = 'softmax')
            accuracy_ = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, axis = 1),tf.cast(label_, tf.int64)), tf.float32), name = 'accuracy')
        return loss_, accuracy_
    def optimizer(self, loss_):
        with tf.name_scope("optimizer"):
            train_step_ = tf.train.AdamOptimizer(self.lr).minimize(loss_, name = 'train_step')
            return train_step_
