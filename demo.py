import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os
import ntpath
import cv2
import math
from PIL import Image, ImageEnhance, ImageOps

img_w, img_h = 256, 128

def autocontrast(image):
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.autocontrast(image, cutoff=1)
    image = np.array(image)[..., ::-1]
    return image

def read_img(image_path):
    img = cv2.imread(image_path)
    img = autocontrast(img)
    img = tf.cast(img, tf.float32) / 255.
    return img

def get_extract_pred_scatter(img, model):
    TILE_SIZE   = 5000 
    PATCH_SIZE  = img_h
    H_STRIDE= 120
    W_STRIDE= img_w
    PATCH_RATE  = 1
    SIZES       = [1, PATCH_SIZE, img_w, 1] 
    # STRIDES     = [1, PATCH_STRIDE, 512, 1] 
    RATES       = [1, PATCH_RATE, PATCH_RATE, 1] 
    PADDING='VALID'
    H, W, C = img.shape

    n_patches = max(int(math.ceil((H / img_h))),2)
    H_STRIDE = (H-img_h) // (n_patches-1)
    H_STRIDE = img_h if not H_STRIDE else H_STRIDE
    n_patches = max(int(math.ceil((W / img_w))),2)
    W_STRIDE = (W-img_w) // (n_patches-1)
    W_STRIDE = img_w if not W_STRIDE else W_STRIDE
    # patch_number
    tile_PATCH_NUMBER = ((H - img_h)//H_STRIDE + 1)*((W - img_w)//W_STRIDE + 1)
    # the indices trick to reconstruct the tile
    x = tf.range(W)
    y = tf.range(H)
    x, y = tf.meshgrid(x, y)
    indices = tf.stack([y, x], axis=-1)
    # making patches, TensorShape([2, 17, 17, 786432])
    img_patches = tf.image.extract_patches(images=tf.expand_dims(
        img, axis=0),     sizes=SIZES, strides=[1, H_STRIDE, W_STRIDE, 1] , rates=RATES, padding=PADDING)
    ind_patches = tf.image.extract_patches(images=tf.expand_dims(
        indices, axis=0), sizes=SIZES, strides=[1, H_STRIDE, W_STRIDE, 1] , rates=RATES, padding=PADDING)
    # squeezing the shape (removing dimension of size 1)
    img_patches = tf.squeeze(img_patches)
    ind_patches = tf.squeeze(ind_patches)
    # reshaping
    img_patches = tf.reshape(
        img_patches, [tile_PATCH_NUMBER, PATCH_SIZE, img_w, C])
    ind_patches = tf.reshape(
        ind_patches, [tile_PATCH_NUMBER, PATCH_SIZE, img_w, 2])
    # Now predict
    pred_patches = model.predict(img_patches, batch_size=50)
    # stitch together the patch summing the overlapping patches probabilities
    pred_tile = tf.scatter_nd(
        indices=ind_patches, updates=pred_patches, shape=(H, W, 2))
    return pred_tile

def display_multiple_img(images, name, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.savefig( name,dpi=400)

def inference(path,model):
    name=ntpath.basename(path)
    orig_img = cv2.imread(path, 1)
    orig_img = autocontrast(orig_img)
    orig_img = orig_img[:,:,::-1]

    img = read_img(path)

    predsTrain = get_extract_pred_scatter(img, model)
    out=np.squeeze(predsTrain)
    out = np.argmax(out, axis=2) * 255
    out = out.astype('uint8')
    
    cv2.imwrite('out/__'+ name+".png",out)
    images = {'Image':orig_img,"Predicted Mask":out}
    display_multiple_img(images,'out/'+ name+".png", 2, 1)

import sys

def main():
    # mpath= sys.argv[1]
    mpath='trained/final-seg4'
    model = tf.keras.models.load_model(mpath,compile=False)

    for ix,path in enumerate(glob('imgs/*')):
        print(path)
        inference(path,model)

if __name__ == "__main__":
    main()