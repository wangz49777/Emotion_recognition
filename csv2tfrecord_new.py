from tqdm import tqdm
import numpy as np
import os
import csv
import tensorflow as tf

channel = 1
default_height = 48
default_width = 48
data_folder_name = 'data'
dataset_folder = 'FER'
csv_file_name = 'fer2013.csv'
csv_new = 'fer2013new.csv'
record_name_train = 'fer2013_train_new.tfrecord'
record_name_test = 'fer2013_test_new.tfrecord'
record_name_val = 'fer2013_val_new.tfrecord'
csv_path = os.path.join(data_folder_name, dataset_folder, csv_file_name)
csv_new_path = os.path.join(data_folder_name, dataset_folder, csv_new)
record_path_train = os.path.join(data_folder_name, dataset_folder, record_name_train)
record_path_test = os.path.join(data_folder_name, dataset_folder, record_name_test)
record_path_val = os.path.join(data_folder_name, dataset_folder, record_name_val)

with open(csv_path, 'r') as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]
    trn = [row[1] for row in rows if row[-1] == 'Training']
    val = [row[1] for row in rows if row[-1] == 'PublicTest']
    tst = [row[1] for row in rows if row[-1] == 'PrivateTest']

with open(csv_new_path, 'r') as f_new:
    csvr_new = csv.reader(f_new)
    header = next(csvr_new)
    rows = [row for row in csvr_new]
    trn_new = [row[2:] for row in rows if row[0] == 'Training']
    val_new = [row[2:] for row in rows if row[0] == 'PublicTest']
    tst_new = [row[2:] for row in rows if row[0] == 'PrivateTest']

def int64_feature(value_):
    return tf.train.Feature(int64_list=tf.train.Int64List(value = value_))

def write_binary(record_name_, images_, labels_, height_ = default_height, width_ = default_width):
    assert len(images_)==len(labels_), 'images_ and labels_ are not the same length'
    writer_ = tf.python_io.TFRecordWriter(record_name_)
    i=0
    for label_ ,image_ in tqdm(zip(labels_, images_)):        
        if label_[-1] == '10':
           continue    
        label_ = np.asarray([int(element) for element in label_[:-2]])        
        image_ = np.asarray([int(element) for element in image_.split()])
        tfrecord_ = tf.train.Example(features = tf.train.Features(feature = {
            "image/label": int64_feature(label_),
            "image/height": int64_feature([height_]),
            "image/width": int64_feature([width_]),
            "image/raw": int64_feature(image_)}))
        writer_.write(tfrecord_.SerializeToString())
    writer_.close()

def main(argv = None):
    write_binary(record_path_train, trn, trn_new)
    write_binary(record_path_test, tst, tst_new)
    write_binary(record_path_val, val, val_new)

if __name__ == '__main__':
    tf.app.run()