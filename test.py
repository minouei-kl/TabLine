import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.layers import *
import math
from glob import glob
from focal_loss import SparseCategoricalFocalLoss, BinaryFocalLoss
from tensorflow.keras.callbacks import TensorBoard

img_h, img_w = 128, 256
batch_size = 8
num_classes = 2
epochs = 4


def load_data(image_path, mask_path):
    img = tf.io.read_file(image_path)

    img = tf.image.decode_jpeg(
        img, channels=3, dct_method='INTEGER_ACCURATE')
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32) / 255.

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.cast(mask, tf.int32)//255

    return img, mask


def make_patches(img, mask):
    image1_patches = tf.image.extract_patches(images=tf.expand_dims(img, 0),
                                              sizes=[1, 128, 256, 1],
                                              strides=[1, 128, 256, 1],
                                              rates=[1, 1, 1, 1],
                                              padding='SAME')[0]
    image1_patch_batch = tf.reshape(image1_patches, (-1, 128, 256, 3))

    mask_patches = tf.image.extract_patches(images=tf.expand_dims(mask, 0),
                                            sizes=[1, 128, 256, 1],
                                            strides=[1, 128, 256, 1],
                                            rates=[1, 1, 1, 1],
                                            padding='SAME')[0]
    mask_patch_batch = tf.reshape(mask_patches, (-1, 128, 256, 1))
    return image1_patch_batch, mask_patch_batch


def make_patches_ds(image, mask):
    return tf.data.Dataset.from_tensor_slices(make_patches(image, mask))


def data_generator(img_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_list, mask_list))
    dataset = dataset.map(
        load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.flat_map(make_patches_ds)

    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


DATA_DIR = "/home/minouei/Downloads/datasets/jcdl-deepfigures-labels/deepfigures-labels/arxiv/ver2/augmented/"

test_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/test/*.jpg')))
test_mask_folder = sorted(
    glob(os.path.join(DATA_DIR, 'annotations/test/*.png')))


test_dataset = data_generator(test_images_folder, test_mask_folder)


class CategoricalF1(tf.keras.metrics.Metric):

    def __init__(self, name="custom_f1", **kwargs):
        super(CategoricalF1, self).__init__(name = name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis = -1)
        y_pred = y_pred[..., tf.newaxis]
        self.precision.update_state(tf.maximum(y_true, 0), y_pred, sample_weight)
        self.recall.update_state(tf.maximum(y_true, 0), y_pred, sample_weight)

    def result(self):
        __prec = self.precision.result()
        __recall = self.recall.result()
        return 2 * __prec * __recall / (__prec + __recall)

class CategoricalMetric(tf.keras.metrics.Metric):

    def __init__(self, metric, name="custom_metric", **kwargs):
        super(CategoricalMetric, self).__init__(name = name, **kwargs)
        self.m = metric

    def reset_states(self):
        self.m.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis = -1)
        y_pred = y_pred[..., tf.newaxis]
        self.m.update_state(tf.maximum(y_true, 0), y_pred, sample_weight)

    def result(self):
        return self.m.result()

import tensorflow.keras.backend as K

def seg_metrics(y_true, y_pred, metric_name, metric_type='standard', drop_last = True, mean_per_class=False, verbose=False):
    """ 
    Compute mean metrics of two segmentation masks, via Keras.
    
    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)
    
    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot 
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.
    
    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """
    
    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')
    
    # always assume one or more classes
    num_classes = K.shape(y_true)[-1]
        
    if not flag_soft:
        # get one-hot encoded masks from y_pred (true masks should already be one-hot)
        y_pred = K.one_hot(K.argmax(y_pred), num_classes)
        y_true = K.one_hot(K.argmax(y_true), num_classes)

    # if already one-hot, could have skipped above command
    # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth)/(mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask =  K.cast(K.not_equal(union, 0), 'float32')
    
    if drop_last:
        metric = metric[:,:-1]
        mask = mask[:,:-1]
    
    if verbose:
        print('intersection, union')
        print(K.eval(intersection), K.eval(union))
        print(K.eval(intersection/union))
    
    # return mean metrics: remaining axes are (batch, classes)
    if flag_naive_mean:
        return K.mean(metric)

    # take mean only over non-absent classes
    class_count = K.sum(mask, axis=0)
    non_zero = tf.greater(class_count, 0)
    non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
    non_zero_count = tf.boolean_mask(class_count, non_zero)
    
    if verbose:
        print('Counts of inputs with class present, metrics for non-absent classes')
        print(K.eval(class_count), K.eval(non_zero_sum / non_zero_count))
        
    return K.mean(non_zero_sum / non_zero_count)

def mean_iou(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via Keras.
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name='iou', **kwargs)

def mean_dice(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via Keras.
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name='dice', **kwargs)

metrics = [

            tf.metrics.SparseCategoricalAccuracy(),
            CategoricalMetric(tf.keras.metrics.Precision(), name = 'custom_precision'),
            CategoricalMetric(tf.keras.metrics.Recall(), name = 'custom_recall'),
            CategoricalF1(name = 'custom_f1'),
            CategoricalMetric(tf.keras.metrics.MeanIoU(num_classes=2),name = 'custom_MeanIoU'),
            CategoricalMetric(tf.keras.metrics.Accuracy(), name = 'custom_accuracy'),

        ]

loss = SparseCategoricalFocalLoss(from_logits=True, gamma=2)
optimizer = tf.keras.optimizers.SGD(lr=1e-2,momentum=0.9, nesterov=True)

mpath='trained/final-seg4'
model = tf.keras.models.load_model(mpath)


model.compile(
    optimizer=optimizer,
    loss=loss, 
    metrics=metrics
    )


eval_out = model.evaluate(test_dataset)
print("Restored model, accuracy: {:5.2f}%, loss: {:5.2f}%".format(
    100 * eval_out[1], eval_out[0]))
print(eval_out)
#loss: 0.0101 - sparse_categorical_accuracy: 0.9842 - custom_tp: 35644484.0000 - custom_fp: 5955177.0000 - custom_tn: 576590016.0000 - custom_fn: 3878334.0000 - custom_accuracy: 0.9842 - custom_precision: 0.8568 - custom_recall: 0.9019 - custom_f1: 0.8788
# 59s 12ms/step
