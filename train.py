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


DATA_DIR = "../ver2/augmented/"

train_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/train/*.jpg')))
train_mask_folder = sorted(glob(os.path.join(DATA_DIR, 'annotations/train/*.png')))

test_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/test/*.jpg')))
test_mask_folder = sorted(glob(os.path.join(DATA_DIR, 'annotations/test/*.png')))

valid_images_folder = sorted(glob(os.path.join(DATA_DIR, 'images/val/*.jpg')))
valid_mask_folder = sorted(glob(os.path.join(DATA_DIR, 'annotations/val/*.png')))

train_dataset = data_generator(train_images_folder, train_mask_folder)
test_dataset = data_generator(test_images_folder, test_mask_folder)
valid_dataset = data_generator(valid_images_folder, valid_mask_folder)


def inception(inputs):

    def bn_relu(X):
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.PReLU()(X)
        return X

    def GCN(layer, f=2, k=15, s=3):
        conv1 = tf.keras.layers.Conv2D(
            f, (k, s), activation=None, padding='same')(layer)
        conv1 = tf.keras.layers.Conv2D(
            f, (s, k), activation=None, padding='same')(conv1)

        conv2 = tf.keras.layers.Conv2D(
            f, (s, k), activation=None, padding='same')(layer)
        conv2 = tf.keras.layers.Conv2D(
            f, (k, s), activation=None, padding='same')(conv2)

        merge = tf.keras.layers.Add()([conv1, conv2])
        return merge

    def inception_residual_block(X, filters, sc=False, suffix=''):
        if sc == True:
            shortcut = X
        s1, s2, s3 = filters/8, filters/4, filters/2

        l1 = tf.keras.layers.Conv2D(s1, 1, 1, padding='same')(X)
        l1 = bn_relu(l1)
        l1 = GCN(l1, k=15)
        l1 = tf.keras.layers.PReLU()(l1)

        l4 = tf.keras.layers.Conv2D(s1, 1, 1, padding='same')(X)
        l4 = bn_relu(l4)
        l4 = GCN(l4, k=45, s=5)
        l4 = tf.keras.layers.PReLU()(l4)

        l2 = tf.keras.layers.Conv2D(s2, 1, 1, padding='same')(X)
        l2 = bn_relu(l2)
        l2 = tf.keras.layers.Conv2D(s2, 7, 1, padding='same')(l2)
        l2 = bn_relu(l2)

        l3 = tf.keras.layers.Conv2D(s3, 1, 1, padding='same')(X)
        l3 = bn_relu(l3)
        l3 = tf.keras.layers.Conv2D(s3, 3, 1, padding='same')(l3)
        l3 = bn_relu(l3)

        X = tf.keras.layers.concatenate([l1, l2, l3, l4])
        # X = tf.keras.layers.concatenate([l1, l2, l4])
        if sc == True:
            X = tf.keras.layers.Concatenate(name='add_'+suffix)([shortcut, X])
        return X

    stem = tf.keras.layers.Conv2D(8, (7, 7), strides=(
        1, 1), padding='same', name='stem1')(inputs)
    stem = bn_relu(stem)
    stem = tf.keras.layers.Conv2D(8, (3, 3), strides=(
        2, 2), padding='same', name='stem2')(stem)
    X = bn_relu(stem)

    X = inception_residual_block(X, 16, suffix='first')
    X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    X = inception_residual_block(X, 16, True, suffix='second')

    model = tf.keras.Model(inputs=inputs, outputs=X,
                           name="mini_inception_resnet")
    return model


def get_model():
    def bn_relu(x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU()(x)
        return x

    def GCN(layer, f=2, k=15, s=3):
        conv1 = tf.keras.layers.Conv2D(
            f, (k, s), activation=None, padding='same')(layer)
        conv1 = tf.keras.layers.Conv2D(
            f, (s, k), activation=None, padding='same')(conv1)

        conv2 = tf.keras.layers.Conv2D(
            f, (s, k), activation=None, padding='same')(layer)
        conv2 = tf.keras.layers.Conv2D(
            f, (k, s), activation=None, padding='same')(conv2)

        merge = tf.keras.layers.Add()([conv1, conv2])
        return merge

    def inception_residual_block(X, filters):
        s1, s2, s3 = filters/8, filters/4, filters/2

        l1 = tf.keras.layers.Conv2D(s1, 1, 1, padding='same')(X)
        l1 = bn_relu(l1)
        l1 = GCN(l1, k=15)
        l1 = tf.keras.layers.PReLU()(l1)

        l4 = tf.keras.layers.Conv2D(s1, 1, 1, padding='same')(X)
        l4 = bn_relu(l4)
        l4 = GCN(l4, k=45, s=5)
        l4 = tf.keras.layers.PReLU()(l4)

        X = tf.keras.layers.concatenate([l1, l4])
        return X

    def decoder(num_classes):
        model_input = tf.keras.Input(shape=(img_h, img_w, 3))

        simplenet = inception(model_input)
        layer = simplenet.get_layer('add_second').output

        layer = tf.keras.layers.Conv2D(8, (1, 1), padding='same',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       use_bias=True)(layer)
        layer = bn_relu(layer)
        br_1 = inception_residual_block(layer, 16)

        layer = tf.keras.layers.Concatenate(axis=-1)([layer, br_1])
        layer = tf.keras.layers.Conv2D(8, (1, 1), padding='same',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       use_bias=True)(layer)
        layer = bn_relu(layer)
        layer = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding="same",
                                                activation="relu", kernel_initializer="glorot_normal")(layer)

        input_b = simplenet.get_layer('concatenate').output
        layer = tf.keras.layers.Concatenate(axis=-1)([input_b, layer])

        layer = tf.keras.layers.Conv2D(8, (1, 1), padding='same',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       use_bias=True)(layer)
        layer = bn_relu(layer)
        br_2 = inception_residual_block(layer, 16)

        layer = tf.keras.layers.Concatenate(axis=-1)([layer, br_2])
        layer = tf.keras.layers.Conv2D(8, (1, 1), padding='same',
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       use_bias=True)(layer)

        layer = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding="same",
                                                activation="relu", kernel_initializer="glorot_normal")(layer)

        box_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1),
                                            padding='same', name='seg_output')(layer)

        return tf.keras.Model(inputs=model_input, outputs=box_output)

    return decoder(num_classes)


model = get_model()

model.summary()
tf.keras.utils.plot_model(model, to_file="seg4.pdf", show_shapes=True, dpi=100)
# from keras_flops import get_flops
# flops = get_flops(model, batch_size=1)
# print(f"FLOPS: {flops / 10 ** 9:.03} G")
# Total params: 860,202
# Trainable params: 859,994
# Non-trainable params: 208
# FLOPS: 0.321 G

# exit()

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

metrics = [
            tf.metrics.SparseCategoricalAccuracy(),
            CategoricalMetric(tf.keras.metrics.Precision(), name = 'custom_precision'),
            CategoricalMetric(tf.keras.metrics.Recall(), name = 'custom_recall'),
            CategoricalF1(name = 'custom_f1'),
            CategoricalMetric(tf.keras.metrics.MeanIoU(num_classes=2),name = 'custom_MeanIoU'),
        ]

loss = SparseCategoricalFocalLoss(from_logits=True, gamma=2)
optimizer = tf.keras.optimizers.SGD(lr=1e-2,momentum=0.9, nesterov=True)

model.compile(
    optimizer=optimizer,
    loss=loss, 
    metrics=metrics)

def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


filepath = 'trained/model-seg/seg4'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='val_loss',
                                                save_best_only=True, verbose=1,
                                                save_weights_only=True,
                                                mode='auto')
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
tensorboard = TensorBoard(
  log_dir='logs',
  histogram_freq=1,
  write_images=True
)

model.fit(train_dataset, epochs=epochs,
          validation_data=valid_dataset,
          callbacks=[checkpoint, lr_callback, tensorboard])


eval_out = model.evaluate(test_dataset)
print("Restored model, accuracy: {:5.2f}%, loss: {:5.2f}%".format(100 * eval_out[1], eval_out[0]))
print(eval_out)


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              loss=loss, metrics=['accuracy'])

write_model_path = 'trained/final-seg4'
model.save(write_model_path)
