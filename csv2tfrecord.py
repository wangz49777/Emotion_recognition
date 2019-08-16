from tqdm import tqdm
from time import sleep
import numpy as np
import os
import csv
import tensorflow as tf

channel = 1
default_height = 48
default_width = 48
data_folder_name = 'data'
csv_file_name = 'fer2013.csv'
record_name_train = 'fer2013_train.tfrecord'
record_name_test = 'fer2013_test.tfrecord'
record_name_eval = 'fer2013_eval.tfrecord'
csv_path = os.path.join(data_folder_name, csv_file_name)
record_path_train = os.path.join(data_folder_name, record_name_train)
record_path_test = os.path.join(data_folder_name, record_name_test)
record_path_eval = os.path.join(data_folder_name, record_name_eval)

with open(csv_path, 'r') as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']

def int64_feature(value_):
    return tf.train.Feature(int64_list=tf.train.Int64List(value = value_))

def write_binary(record_name_, labels_images_, height_ = default_height, width_ = default_width):
    writer_ = tf.python_io.TFRecordWriter(record_name_)
    for label_images_ in tqdm(labels_images_):
        label_ = int(label_images_[0])
        image_ = np.asarray([int(p) for p in label_images_[-1].split()])
        tfrecord_ = tf.train.Example(features = tf.train.Features(feature = {
            "image/label": int64_feature([label_]),
            "image/height": int64_feature([height_]),
            "image/width": int64_feature([width_]),
            "image/raw": int64_feature(image_)}))
        writer_.write(tfrecord_.SerializeToString())
    writer_.close()

def main(argv = None):
    write_binary(record_path_train, trn)
    write_binary(record_path_test, tst)
    write_binary(record_path_eval, val)

if __name__ == '__main__':
    main()
