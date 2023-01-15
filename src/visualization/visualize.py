
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image

def show_feature_maps(activation):
    plt.figure(figsize=(20,20))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(activation[0,:,:,i])
    plt.show()

def tensor_to_img(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def img_to_tensor(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(content_img, style_img, combined_image):
    plt.figure(figsize=[15,15])
    plt.subplot(131);plt.imshow(content_img[:,:,::-1]);plt.title("Content Image");plt.axis('off');
    plt.subplot(132);plt.imshow(style_img[:,:,::-1]);plt.title("Style Image");plt.axis('off');
    plt.subplot(133);plt.imshow(combined_image[:,:,::-1]);plt.title("Combined Image");plt.axis('off');
    