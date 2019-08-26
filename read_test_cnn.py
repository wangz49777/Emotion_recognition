import os
import tensorflow as tf
import numpy as np
import train_val_cnn
os.environ["CUDA_VISIABLE_DEVICES"] = "3"

data_folder_name = 'data'
dataset_folder = 'FER'
test_model_folder = 'testmodel'
ckpt_name = 'emotion_cnn.ckpt'
ckpt_path = os.path.join(test_model_folder, ckpt_name)

emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)

record_test = 'test/test.tfrecord'
test_path = os.path.join(data_folder_name, dataset_folder, record_test)
shuffle_pool_size = 3000
batch_size = 3000

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

        data_set_test = train_val_cnn.get_dataset(test_path)
        data_set_test = data_set_test.shuffle(shuffle_pool_size).batch(batch_size).repeat()
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_test.output_types, data_set_test.output_shapes)
        x_input_bacth, y_target_batch = iterator.get_next()

        #for i in range(10):
        x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: test_handle})
        test_feed_dict = {x_input: x_batch, y_target: y_batch, dropout: 1.0}
        test_accuracy = sess.run(accuracy, test_feed_dict)
        print('Test Acc : {:.3f}. '.format(test_accuracy))

if __name__ == '__main__':
    tf.app.run()
