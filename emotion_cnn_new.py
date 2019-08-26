import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers

class Emotion_cnn():
    def __init__(self):
        with tf.name_scope("parameters"):
            self.class_num = 8 #类别个数
            self.channel = 1 #图像通道个数
            self.hidden_dim = 1024 #全连接层维度
            self.full_shape = 4096 #
            self.optimizer_ = 'Adam' #优化算法
        with tf.name_scope("input"):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.channel], name='x_input') #输入数据
        with tf.name_scope("y_target"):
            self.y_target = tf.placeholder(dtype=tf.int32, shape=[None, self.class_num], name='y_target') #标签
        with tf.name_scope("cnn_structure"):
            self.maxpool_ksize = [1, 2, 2, 1] #池化窗口大小，[batch, height, width, channel]
            self.maxpool_strides = [1, 2, 2, 1] #窗口滑动的步长
            self.padding = "SAME" #补丁类型
            self.activation = "relu" #激活函数
            self.initializer = initializers.xavier_initializer() #变量初始化方法
            self.filter = [[1, 1, 1, 64], [3, 3, 64, 64], [3, 3, 64, 128], [3, 3, 128, 128], [3, 3, 128, 256], [3, 3, 256, 256]] #卷积核尺寸
            self.bias = [64, 128, 256] #偏置
            self.strides = [1, 1, 1, 1] #卷积步长
            self.batch_size = tf.shape(self.x_input)[0] #批数量
        with tf.name_scope("hyperparameters"):
            self.lr = 0.001 #学习速率
            self.conv_dropout = 0.25
            self.fc_dropout = tf.placeholder(dtype=tf.float32, name='dropout') #
            self.depth_radius = 5 #归一化半径？
            self.norm_bias = 2.0 #归一化偏置？
            self.alpha = 1e-3 #？？
            self.beta = 0.75 #？？
#         with tf.name_scope("evaluate"):
#         self.loss, self.accuracy = self.loss_accuracy(self.output_layer(self.cnn_layer()), self.y_target)
    # 定义卷积核
    def weight_variable(self, name, shape):
        with tf.variable_scope("weight"):
            weight = tf.get_variable(name, shape, dtype=tf.float32, initializer=self.initializer)
        return weight
    def bias_variable(self, name, shape):
        with tf.variable_scope("bias"):
            bias = tf.get_variable(name, shape, dtype=tf.float32, initializer=self.initializer)
        return bias
    def cnn_activation(self, input_, weight_, bias_):
        conv = tf.nn.conv2d(input_, weight_, self.strides, self.padding)
        conv_relu = tf.nn.relu(tf.nn.bias_add(conv, bias_))
        return conv_relu
    # 定义卷积层
    def cnn_layer(self):
        with tf.name_scope("conv1"):
            conv1_1_weight = self.weight_variable('conv1_1_weight', self.filter[0])
            conv1_1_bias = self.bias_variable('conv1_1_bias', self.bias[0])
            conv1_2_weight = self.weight_variable('conv1_2_weight', self.filter[1])
            conv1_2_bias = self.bias_variable('conv1_2_bias', self.bias[0])
            conv1_1 = self.cnn_activation(self.x_input, conv1_1_weight, conv1_1_bias)
            conv1_2 = self.cnn_activation(conv1_1, conv1_2_weight, conv1_2_bias)
            pool1 = tf.nn.max_pool(conv1_2, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer1')
            dropout1 = tf.nn.dropout(pool1, keep_prob = self.conv_dropout)

        with tf.name_scope("conv2"):
            conv2_1_weight = self.weight_variable('conv2_1_weight', self.filter[2])
            conv2_1_bias = self.bias_variable('conv2_1_bias', self.bias[1])
            conv2_2_weight = self.weight_variable('conv2_2_weight', self.filter[3])
            conv2_2_bias = self.bias_variable('conv2_2_bias', self.bias[1])
            conv2_1 = self.cnn_activation(dropout1, conv2_1_weight, conv2_1_bias)
            conv2_2 = self.cnn_activation(conv2_1, conv2_2_weight, conv2_2_bias)

            pool2 = tf.nn.max_pool(conv2_2, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer2')
            dropout2 = tf.nn.dropout(pool2, keep_prob = self.conv_dropout)
        with tf.name_scope("conv3"):
            conv3_1_weight = self.weight_variable('conv3_1_weight', self.filter[4])
            conv3_1_bias = self.bias_variable('conv3_1_bias', self.bias[2])
            conv3_2_weight = self.weight_variable('conv3_2_weight', self.filter[5])
            conv3_2_bias = self.bias_variable('conv3_2_bias', self.bias[2])
            conv3_3_weight = self.weight_variable('conv3_3_weight', self.filter[5])
            conv3_3_bias = self.bias_variable('conv3_3_bias', self.bias[2])
            conv3_1 = self.cnn_activation(dropout2, conv3_1_weight, conv3_1_bias)
            conv3_2 = self.cnn_activation(conv3_1, conv3_2_weight, conv3_2_bias)
            conv3_3 = self.cnn_activation(conv3_2, conv3_3_weight, conv3_3_bias)
            pool3 = tf.nn.max_pool(conv3_3, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer3')
            dropout3 = tf.nn.dropout(pool3, keep_prob = self.conv_dropout)
        with tf.name_scope("conv4"):
            conv4_1_weight = self.weight_variable('conv4_1_weight', self.filter[5])
            conv4_1_bias = self.bias_variable('conv4_1_bias', self.bias[2])
            conv4_2_weight = self.weight_variable('conv4_2_weight', self.filter[5])
            conv4_2_bias = self.bias_variable('conv4_2_bias', self.bias[2])
            conv4_3_weight = self.weight_variable('conv4_3_weight', self.filter[5])
            conv4_3_bias = self.bias_variable('conv4_3_bias', self.bias[2])
            conv4_1 = self.cnn_activation(dropout3, conv4_1_weight, conv4_1_bias)
            conv4_2 = self.cnn_activation(conv4_1, conv4_2_weight, conv4_2_bias)
            conv4_3 = self.cnn_activation(conv4_2, conv4_3_weight, conv4_3_bias)
            pool4 = tf.nn.max_pool(conv4_3, self.maxpool_ksize, self.maxpool_strides, self.padding, name = 'pool_layer4')
            dropout4 = tf.nn.dropout(pool4, keep_prob = self.conv_dropout)
        return dropout4
    #输出层，全连接+softmax
    def output_layer(self, fc_input):
        with tf.name_scope("output_layer"):
            with tf.name_scope("fully_connected_layer1"):

                fc_input = tf.reshape(fc_input, [self.batch_size, -1])
                fc1_weight = self.weight_variable('fc1_weight', [self.full_shape, self.hidden_dim])
                fc1_bias = self.bias_variable('fc1_bias', [self.hidden_dim])
                fc1 = tf.add(tf.matmul(fc_input, fc1_weight), fc1_bias)

                dropout1 = tf.nn.dropout(tf.nn.relu(fc1), keep_prob = self.fc_dropout)
            with tf.name_scope("fully_connected_layer2"):

                fc2_weight = self.weight_variable('fc2_weight', [self.hidden_dim, self.hidden_dim])
                fc2_bias = self.bias_variable('fc2_bias', [self.hidden_dim])
                fc2 = tf.add(tf.matmul(dropout1, fc2_weight), fc2_bias)

                output2 = tf.nn.dropout(tf.nn.relu(fc2), keep_prob = self.fc_dropout)
            with tf.name_scope("softmax_layer"):
                out_weight = self.weight_variable('out_weight', [self.hidden_dim, self.class_num])
                out_bias = self.bias_variable('out_bias', [self.class_num])
                fc_output = tf.add(tf.matmul(output2, out_weight), out_bias, name = 'fc_output')
                softmax = tf.nn.softmax(fc_output)
        return softmax
    #损失函数和准确率
    def loss_accuracy(self, logits_, labels_):
        with tf.name_scope("loss"):

            labels_ = tf.nn.softmax(tf.cast(labels_, tf.float32))
            labels_ = tf.argmax(labels_, axis = 1)
#             loss_ = -tf.reduce_sum(tf.cast(label_, tf.float32) * tf.log(logits_))
#         return loss_
            loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_, labels = labels_), name = 'cross_entropy_loss')
        with tf.name_scope("evaluate"):
#             equal = tf.cast(tf.equal(tf.argmax(logits_, axis= 1), label), tf.float32)
#             accuracy_ = tf.reduce_mean(equal, name = 'accuracy')
#         return loss_, accuracy_
#             softmax = tf.nn.softmax(logits_, name = 'softmax')
            accuracy_ = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_, axis = 1),tf.cast(labels_, tf.int64)), tf.float32), name = 'accuracy')
        return loss_, accuracy_
    #优化方法
    def optimizer(self, loss_):
        with tf.name_scope("optimizer"):
            train_step_ = tf.train.AdamOptimizer(self.lr).minimize(loss_, name = 'train_step')
            return train_step_
