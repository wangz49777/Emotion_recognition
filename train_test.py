import tensorflow as tf
import os
import numpy as np
import emotion_cnn_8
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.flags
FLAGS = flags.FLAGS
# FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_integer("batch_size", 256, "The size of a mini-batch")
flags.DEFINE_integer("val_batch_size", 2000, "The size of a val mini-batch")
flags.DEFINE_integer("test_batch_size", 3000, "The size of a test mini-batch")
flags.DEFINE_integer("shuffle_pool_size", 4000, "The size of shuffle pool")
flags.DEFINE_integer("max_step", 200, "train max step")
flags.DEFINE_string("dataset_path", "data/FER/", "dataset path") #数据集路径
flags.DEFINE_string("output_path", "data/model", "model save path") #保存训练时生成的模型文件
flags.DEFINE_string("mode", None, "train or test") #训练或测试
flags.DEFINE_string("test_model_folder", "testmodel", "test model path") #测试时读取模型文件路径


channel = 1  # 图像通道数
default_height = 48  # 图像宽高
default_width = 48
class_num = 8
# shuffle_pool_size = 4000 # 内存小就调小些
# retrain = False # 是否要继续之前的训练
# save_ckpt_name = 'emotion_cnn.ckpt' #模型保存文件
ckpt_name = 'emotion_cnn.ckpt'
train_file = os.path.join(FLAGS.dataset_path, 'train', 'train.tfrecord')
val_file = os.path.join(FLAGS.dataset_path, 'val', 'val.tfrecord')
test_file = os.path.join(FLAGS.dataset_path, 'test', 'test.tfrecord')
model_save_path = os.path.join(FLAGS.output_path, ckpt_name)
tensorboard_path = os.path.join(FLAGS.output_path, 'tensorboard')
ckpt_path = os.path.join(FLAGS.test_model_folder, ckpt_name)
# test_batch_size = 3000

#数据增强
def pre_process_img(image):
    image = tf.image.random_flip_left_right(image) #按水平 (从左向右) 随机翻转图像
    image = tf.image.random_brightness(image, max_delta=32./255) #通过随机因子调整图像的亮度
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2) #通过随机因子调整图像的对比度
    image = tf.random_crop(image, [default_height-np.random.randint(0, 4), default_width-np.random.randint(0, 4), 1]) #随机地将张量裁剪为给定的大小
    image = tf.image.resize_images(image, [default_height, default_width]) #使用指定的method调整images为size
    return image
#读入tfrecord数据
#解析feature信息
def parse_function(serial_example):
    dics ={"image/label": tf.FixedLenFeature([class_num], tf.int64),
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
#     image_ = tf.image.resize_images(image_, [default_height, default_width])
    image_ = pre_process_img(image_)
    return image_, label_
def get_dataset(record_name_):
    data_set_ = tf.data.TFRecordDataset(record_name_)
    return data_set_.map(parse_function)
def time_():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def main(argv):
    if FLAGS.mode == 'train':

        data_set_train = get_dataset(train_file) #读取训练数据
        data_set_train = data_set_train.shuffle(FLAGS.shuffle_pool_size).batch(FLAGS.batch_size).repeat() #打乱，分批
        data_set_train_iter = data_set_train.make_one_shot_iterator()

        data_set_val = get_dataset(val_file)
        data_set_val = data_set_val.shuffle(FLAGS.shuffle_pool_size).batch(FLAGS.batch_size).repeat()
        data_set_val_iter = data_set_val.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types, data_set_train.output_shapes)
        x_input_bacth, y_target_batch = iterator.get_next()

        cnn_model = emotion_cnn_8.Emotion_cnn()
        x_input = cnn_model.x_input
        y_target = cnn_model.y_target
        cnn_output = cnn_model.output_layer(cnn_model.cnn_layer())
        loss, accuracy = cnn_model.loss_accuracy(cnn_output, y_target)
        train_step = cnn_model.optimizer(loss)
        dropout = cnn_model.dropout

        with tf.name_scope('Loss_and_Accuracy'):
            tf.summary.scalar('Loss', loss)
            tf.summary.scalar('Accuracy', accuracy)
            summary_op = tf.summary.merge_all() #统计loss，accuracy

        saver = tf.train.Saver(max_to_keep=1)
        max_accuracy = 0
        gpu_options = tf.GPUOptions(allow_growth = True) #按需求分配GPU内存
        config = tf.ConfigProto(gpu_options = gpu_options)
        with tf.Session(config = config) as sess:
            summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
            sess.run(tf.global_variables_initializer()) #初始化变量
            train_handle = sess.run(data_set_train_iter.string_handle())
            val_handle = sess.run(data_set_val_iter.string_handle())

            for i in range(1, FLAGS.max_step + 1):
                x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: train_handle})
                y_label = np.argmax(y_batch, axis= 1)
                train_feed_dict = {x_input: x_batch, y_target: y_label, dropout: 0.5}
                # train_feed_dict = {x_input: x_batch, y_target: y_batch, dropout: 0.5}
                sess.run(train_step, train_feed_dict)
                if i % 50 == 0:
                    train_loss, train_accuracy = sess.run([loss, accuracy], train_feed_dict)
                    print('{} : Generation # {}. Train Loss : {:.3f} . '  'Train Acc : {:.3f}. '.format(time_(), i, train_loss, train_accuracy))
                    summary_writer.add_summary(sess.run(summary_op, train_feed_dict), i)
                if i % 100 == 0:
                    val_x_batch, val_y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: val_handle})
                    y_label = np.argmax(y_batch, axis= 1)
                    val_feed_dict = {x_input: x_batch, y_target: y_label, dropout: 1.0}
                    # val_feed_dict = {x_input: val_x_batch, y_target: val_y_batch, dropout: 1.0}
                    val_loss, val_accuracy = sess.run([loss, accuracy], val_feed_dict)
                    print('{} : Generation # {}. val Loss : {:.3f} . '  'val Acc : {:.3f}. '.format(time_(), i, val_loss, val_accuracy))
                    #summary_writer.add_summary(sess.run(summary_op, train_feed_dict), i)
                    if val_accuracy >= max_accuracy and i > FLAGS.max_step / 2:
                        max_accuracy = val_accuracy
                        # saver.save(sess, os.path.join(model_save_path, save_ckpt_name))
                        saver.save(sess, os.path.join(model_save_path))
                        print('{} : Generation # {}. --model saved--'.format(time_(), i))
            print('{} : Last accuracy : {}'.format(time_(), max_accuracy))
    elif FLAGS.mode == 'test':
        gpu_options = tf.GPUOptions(allow_growth = True)
        config = tf.ConfigProto(gpu_options = gpu_options)
        with tf.Session(config = config) as sess:
            saver = tf.train.import_meta_graph(ckpt_path + '.meta')
            saver.restore(sess, ckpt_path)
            graph = tf.get_default_graph()

            x_input = graph.get_tensor_by_name('input/x_input:0')
            y_target = graph.get_tensor_by_name('y_target/y_target:0')
            dropout = graph.get_tensor_by_name('hyperparameters/dropout:0')
            #softmax = graph.get_tensor_by_name('evaluate/softmax:0')
            accuracy = graph.get_tensor_by_name('evaluate/accuracy:0')

            data_set_test = get_dataset(test_file)
            data_set_test = data_set_test.shuffle(FLAGS.shuffle_pool_size).batch(FLAGS.test_batch_size).repeat()
            data_set_test_iter = data_set_test.make_one_shot_iterator()
            test_handle = sess.run(data_set_test_iter.string_handle())

            handle = tf.placeholder(tf.string, shape=[], name='handle')
            iterator = tf.data.Iterator.from_string_handle(handle, data_set_test.output_types, data_set_test.output_shapes)
            x_input_bacth, y_target_batch = iterator.get_next()

            #for i in range(10):
            x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: test_handle})
            y_label = np.argmax(y_batch, axis= 1)
            test_feed_dict = {x_input: x_batch, y_target: y_label, dropout: 1.0}
            # test_feed_dict = {x_input: x_batch, y_target: y_batch, dropout: 1.0}
            test_accuracy = sess.run(accuracy, test_feed_dict)
            print('Test Acc : {:.3f}. '.format(test_accuracy))
    else:
        print('select the mode : train or test')

if __name__ == '__main__':
    tf.app.run()
