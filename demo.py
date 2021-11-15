import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import os
import ntpath
import cv2
import math
from PIL import Image, ImageEnhance, ImageOps

img_w, img_h = 512, 128
batch_size = 4
num_classes = 2

def autocontrast(image):
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.autocontrast(image, cutoff=1)
    image = np.array(image)[..., ::-1]
    return image

# def read_img(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(
#         img, channels=3, dct_method='INTEGER_ACCURATE')
#     img.set_shape([None, None, 3])
#     img = tf.cast(img, tf.float32) / 255.
#     return img

def read_img(image_path):
    img = cv2.imread(image_path)
    # (h, w) = img.shape[:2]
    # if h<128:
    #     r = 128 / float(h)
    #     img = cv2.resize(img, (int(w * r), 128), interpolation = cv2.INTER_AREA)

    img = autocontrast(img)
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # img = cv2.cvtColor(lab_planes[0], cv2.COLOR_GRAY2BGR)
    img = tf.cast(img, tf.float32) / 255.
    return img

def get_extract_pred_scatter(img, model):
    TILE_SIZE   = 5000 
    PATCH_SIZE  = 128
    H_STRIDE= 120
    W_STRIDE= 512
    PATCH_RATE  = 1
    SIZES       = [1, PATCH_SIZE, 512, 1] 
    # STRIDES     = [1, PATCH_STRIDE, 512, 1] 
    RATES       = [1, PATCH_RATE, PATCH_RATE, 1] 
    PADDING='VALID'
    H, W, C = img.shape

    n_patches = max(int(math.ceil((H / 128))),2)
    H_STRIDE = (H-128) // (n_patches-1)
    H_STRIDE = 128 if not H_STRIDE else H_STRIDE
    n_patches = max(int(math.ceil((W / 512))),2)
    W_STRIDE = (W-512) // (n_patches-1)
    W_STRIDE = 512 if not W_STRIDE else W_STRIDE
    # patch_number
    tile_PATCH_NUMBER = ((H - 128)//H_STRIDE + 1)*((W - 512)//W_STRIDE + 1)
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
        img_patches, [tile_PATCH_NUMBER, PATCH_SIZE, 512, C])
    ind_patches = tf.reshape(
        ind_patches, [tile_PATCH_NUMBER, PATCH_SIZE, 512, 2])
    # Now predict
    pred_patches = model.predict(img_patches, batch_size=50)
    # stitch together the patch summing the overlapping patches probabilities
    pred_tile = tf.scatter_nd(
        indices=ind_patches, updates=pred_patches, shape=(H, W, 2))
    return pred_tile



# model = tf.keras.models.load_model('trained/final-sec3',compile=False)

# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
#               loss=loss, metrics=['accuracy'])
# write_model_path= 'boder1_640'
# model.save(write_model_path)
def display_multiple_img(images, name, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    # plt.show()
    plt.savefig( name,dpi=400)


def inference(path,model):
    name=ntpath.basename(path)
    orig_img = cv2.imread(path, 1)
    # gt_img = cv2.imread(path.replace("imgs","gt"), 1)
    orig_img = autocontrast(orig_img)
    orig_img = orig_img[:,:,::-1]

    img = read_img(path)

    # gaussian_3 = cv2.GaussianBlur(orig_img, (0, 0), 2.0)
    # orig_img = cv2.addWeighted(orig_img, 1.5, gaussian_3, -0.5, 0, orig_img)
    predsTrain = get_extract_pred_scatter(img, model)
    # predsTrain = model.predict(np.expand_dims((img),axis=0))
    out=np.squeeze(predsTrain)
    out = np.argmax(out, axis=2) * 255
    out = out.astype('uint8')
    
    # out2, bboxes = postp(out,orig_img)
    # out = cv2.resize(out, orig_img.shape[1::-1],interpolation=cv2.INTER_CUBIC)
    # bboxes = cv2.resize(bboxes, orig_img.shape[1::-1],interpolation=cv2.INTER_CUBIC)
    # plt.subplot(2, 1, 1)
    # plt.imshow(orig_img)
    # plt.title("image")
    # plt.subplot(2, 1, 2)
    # plt.imshow(out)
    # plt.title("Predicted Mask")
    # plt.savefig( 'out/'+ name+".png",dpi=400)
    cv2.imwrite('out/__'+ name+".png",out)

    images = {'Image':orig_img,"Predicted Mask":out}#,"ground Truth":gt_img}

    display_multiple_img(images,'out/'+ name+".png", 2, 1)




import sys

def main():
    # mpath= sys.argv[1]
    mpath='trained/final-seg4'
    model = tf.keras.models.load_model(mpath,compile=False)

    for ix,path in enumerate(glob('imgs/*')):
    # for ix,path in enumerate(glob(os.path.join(DATA_DIR, 'images/val/*.jpg'))):
        print(path)
        inference(path,model)

    # path=os.path.join(DATA_DIR, 'images/val/765_eur0_0980_00_2.jpg')
    # inference(path)

if __name__ == "__main__":
    main()