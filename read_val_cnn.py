import os
import tensorflow as tf
import numpy as np
import train_test_cnn
os.environ["CUDA_VISIABLE_DEVICES"] = "3"

data_folder_name = 'data/'
ckpt_name = 'cnn_emotion_classifier.ckpt'
ckpt_path = os.path.join(data_folder_name, ckpt_name)

emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)

record_name_val = data_folder_name + 'fer2013_eval.tfrecord'
shuffle_pool_size = 1000
batch_size = 256

def main(argv):
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(gpu_options = gpu_options)
    with tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()

        x_input = graph.get_tensor_by_name('input/x_input:0')
        y_target = graph.get_tensor_by_name('y_target/y_target:0')
        dropout = graph.get_tensor_by_name('hyperparameters/dropout:0')
        softmax = graph.get_tensor_by_name('evaluate/softmax:0')
        accuracy = graph.get_tensor_by_name('evaluate/accuracy:0')

        data_set_val = train_test_cnn.get_dataset(record_name_val)
        data_set_val = data_set_val.shuffle(shuffle_pool_size).batch(batch_size).repeat()
        data_set_val_iter = data_set_val.make_one_shot_iterator()
        val_handle = sess.run(data_set_val_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_val.output_types, data_set_val.output_shapes)
        x_input_bacth, y_target_batch = iterator.get_next()

        for i in range(10):
            x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: val_handle})
            val_feed_dict = {x_input: x_batch, y_target: y_batch, dropout: 1.0}
            val_accuracy = sess.run(accuracy, val_feed_dict)
            print('Generation # {}. '  'Train Acc : {:.3f}. '.format(i, val_accuracy))

if __name__ == '__main__':
    tf.app.run()
