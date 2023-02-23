import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

model=hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :] #we are making sure that our image is inside of a new array
    #so we are passing through tf.newaxis and then we are passing through an ins
    return img


content_image = load_image('.jpg')
style_image = load_image('.jpg')


content_image.shape

plt.imshow(np.squeeze(style_image))
plt.show()


stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

plt.imshow(np.squeeze(stylized_image))
plt.show()

cv2.imwrite('generated_img.jpg', cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))