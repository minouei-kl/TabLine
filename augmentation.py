from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import os
import cv2
import glob
from scipy.ndimage.filters import median_filter
import ntpath


def convert(im, msk):
    seq = iaa.Sequential([
        # GaussianBlur and MotionBlur
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.5))),
        iaa.Sometimes(0.3, iaa.MotionBlur()),
        iaa.Sometimes(0.3, iaa.AverageBlur(k=(1, 3))),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.45, 1.25)),
        iaa.Sometimes(0.3,iaa.AddToBrightness((-30, 30))),
        # Simulate shadow distortion
        iaa.Sometimes(0.3,
                      iaa.BlendAlphaFrequencyNoise(
                          exponent=-4.0,
                          foreground=iaa.Multiply(iap.Choice(
                              [0.7, 1.3]), per_channel=False),
                          size_px_max=32,
                          upscale_method="linear",
                          iterations=1,
                          sigmoid=True
                      )),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(
            0.0, 0.05*255), per_channel=0.5),
        # iaa.ImpulseNoise(),
        iaa.SaltAndPepper(0.001),
        # iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Sometimes(0.7, iaa.Multiply((0.7, 1.3), per_channel=True)),
        iaa.MultiplyHueAndSaturation((0.7, 1.3), per_channel=True),
        iaa.MultiplyHueAndSaturation((0.6, 1.4), per_channel=True),
        # iaa.AddToHue((-150, 150)),
        # iaa.Sometimes(0.5, iaa.ChangeColorTemperature((1100, 2000))),
        # Apply affine transformations to each image.
        iaa.PerspectiveTransform(scale=(0.001, 0.01)),
        iaa.Sometimes(0.7, iaa.PiecewiseAffine(
            scale=(0.0021, 0.0042), cval=240)),
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq(image=im, segmentation_maps=msk)
    return images_aug


def resize(out, mo):
    h,w,c = out.shape
    hr = h /128
    if hr < 1:
        f_h=128
    else:
        x=(hr/10)-1
        f_h=h// (2+x)
        f_h=max(128,f_h)
    f_w=(f_h/h)*w
    if f_w <256:
        f_w=256
        f_h=(f_w/w)*h
    out=cv2.resize(out, (int(f_w),int(f_h)), interpolation= cv2.INTER_AREA)
    mo=cv2.resize(mo, (int(f_w),int(f_h)), interpolation= cv2.INTER_AREA)
    ret, mo = cv2.threshold(mo, 50, 255, cv2.THRESH_BINARY)
    return out, mo


root = '/home/minouei/Downloads/datasets/jcdl-deepfigures-labels/deepfigures-labels/arxiv/ver2/raw/images/'
splits = ['train','test','val']
for split in splits:
    for full_path in glob.glob(os.path.join(root, split, '*.jpg')):
        print(full_path)
        name = ntpath.basename(full_path)
        im = cv2.imread(full_path)

        mask_path = full_path.replace(
            'jpg', 'png').replace('images', 'annotations')
        msk = cv2.imread(mask_path, 0)

        label = SegmentationMapsOnImage(msk, shape=msk.shape)
        out, mo = convert(im, label)
        mask_out = mo.get_arr()
        
        out_path = os.path.join('images',split, name)

        out, mask_out = resize(out, mask_out)
        cv2.imwrite(out_path, out)
        cv2.imwrite(out_path.replace('jpg', 'png').replace('images', 'annotations'), mask_out)
