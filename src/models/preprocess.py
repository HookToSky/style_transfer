import cv2
import tensorflow as tf

from os import path, sep

def resize_img(path_to_img):
    image = cv2.imread(path_to_img)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

    normalized_path = path.normpath(path_to_img)
    path_components = normalized_path.split(sep)
    cv2.imwrite(f"../{path_components[1]}/interim/{path_components[3]}/{path_components[-1]}",  image)

    return image

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var