import tensorflow as tf
import os
import numpy as np
import emotion_cnn
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

channel = 1  # 图像通道数
default_height = 48  # 图像宽高
default_width = 48
batch_size = 256  # 批尺寸，内存小就调小些
test_batch_size = 256  # 测试时的批尺寸，内存小就调小些
shuffle_pool_size = 4000 # 内存小就调小些
generations = 60  # 总迭代数
retrain = False # 是否要继续之前的训练
data_folder_name = 'data/'
model_path = 'model'
record_name_train = data_folder_name + 'fer2013_train.tfrecord'
record_name_test = data_folder_name + 'fer2013_test.tfrecord'
# record_name_eval = data_folder_name + 'fer2013_eval.tfrecord'
save_ckpt_name = 'cnn_emotion_classifier.ckpt'
model_log_name = 'model_log.txt'
tensorboard_name = 'tensorboard'
tensorboard_path = os.path.join(data_folder_name, tensorboard_name)
model_log_path = os.path.join(data_folder_name, model_path, model_log_name)
#数据增强
def pre_process_img(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32./255)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.random_crop(image, [default_height-np.random.randint(0, 4), default_width-np.random.randint(0, 4), 1])
    image = tf.image.resize_images(image, [default_height, default_width])
    return image
#读入tfrecord数据
#解析feature信息
def parse_function(serial_example):
    dics ={"image/label": tf.FixedLenFeature([], tf.int64),
           "image/height": tf.FixedLenFeature([], tf.int64),
           "image/width": tf.FixedLenFeature([], tf.int64),
           "image/raw": tf.FixedLenFeature([default_width*default_height*channel], tf.int64)}
    features_ = tf.parse_single_example(serial_example, dics)
    label_ = tf.cast(features_["image/label"], tf.int32)
    height_ = tf.cast(features_["image/height"], tf.int32)
    width_ = tf.cast(features_["image/width"], tf.int32)
    image_ = tf.cast(features_["image/raw"], tf.int32)
    image_ = tf.reshape(image_, [height_, width_, channel])
    image_ = tf.multiply(tf.cast(image_, tf.float32), 1. / 255)
    image_ = tf.image.resize_images(image_, [default_height, default_width])
    return image_, label_
def get_dataset(record_name_):
    data_set_ = tf.data.TFRecordDataset(record_name_)
    return data_set_.map(parse_function)
def time_():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def main(argv):
    data_set_train = get_dataset(record_name_train)
    data_set_train = data_set_train.shuffle(shuffle_pool_size).batch(batch_size).repeat()
    data_set_train_iter = data_set_train.make_one_shot_iterator()

    data_set_test = get_dataset(record_name_test)
    data_set_test = data_set_test.shuffle(shuffle_pool_size).batch(batch_size).repeat()
    data_set_test_iter = data_set_test.make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[], name='handle')
    iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types, data_set_train.output_shapes)
    x_input_bacth, y_target_batch = iterator.get_next()

    cnn_model = emotion_cnn.Emotion_cnn()
    x_input = cnn_model.x_input
    y_target = cnn_model.y_target
    cnn_output = cnn_model.output_layer(cnn_model.cnn_layer())
    loss, accuracy = cnn_model.loss_accuracy(cnn_output, y_target)
    train_step = cnn_model.optimizer(loss)
    dropout = cnn_model.dropout

    with tf.name_scope('Loss_and_Accuracy'):
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=1)
    max_accuracy = 0
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(gpu_options = gpu_options)
    with tf.Session(config = config) as sess:
        summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        train_handle = sess.run(data_set_train_iter.string_handle())
        test_handle = sess.run(data_set_test_iter.string_handle())

        for i in range(1, generations + 1):
            x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: train_handle})
            train_feed_dict = {x_input: x_batch, y_target: y_batch, dropout: 0.5}
            sess.run(train_step, train_feed_dict)
            if i % 10 == 0:
                train_loss, train_accuracy = sess.run([loss, accuracy], train_feed_dict)
                print('{} : Generation # {}. Train Loss : {:.3f} . '  'Train Acc : {:.3f}. '.format(time_(), i, train_loss, train_accuracy))
                summary_writer.add_summary(sess.run(summary_op, train_feed_dict), i)
            if i % 20 == 0:
                test_x_batch, test_y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: test_handle})
                test_feed_dict = {x_input: test_x_batch, y_target: test_y_batch, dropout: 1.0}
                test_loss, test_accuracy = sess.run([loss, accuracy], test_feed_dict)
                print('{} : Generation # {}. Test Loss : {:.3f} . '  'Test Acc : {:.3f}. '.format(time_(), i, test_loss, test_accuracy))
                summary_writer.add_summary(sess.run(summary_op, train_feed_dict), i)
                if test_accuracy >= max_accuracy and i > generations / 2:
                    max_accuracy = test_accuracy
                    saver.save(sess, os.path.join(data_folder_name, save_ckpt_name))
                    print('{} : Generation # {}. --model saved--'.format(time_(), i))
        print('Last accuracy : ', max_accuracy)

if __name__ == '__main__':
    tf.app.run()