#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
        encoded_data_io = io.BytesIO(encoded_data)
        image = PIL.Image.open(encoded_data_io)
        if image.format != 'JPG':
            raise ValueError('Image format not JPG')
        key = hashlib.sha256(encoded_data).hexdigest()

    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()
    encoded_label_io = io.BytesIO(encoded_label)
    mask = PIL.Image.open(encoded_label_io)
    if mask.format != 'PNG':
        raise ValueError('Mask format not PNG')

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': None,
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': None,
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here
    writer = tf.python_io.TFRecordWriter(output_filename)
    data_label=list(zip(*file_pars))
    data_path=data_label[0]
    label_path=data_label[1]
    for i in data_path:
        try:
            tf_example=dict_to_tf_example(data_path[i],label_path[i])
            if tf_example:
                writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', label_path[i])
    writer.close()


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
