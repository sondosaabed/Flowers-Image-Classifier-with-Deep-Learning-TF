import numpy as np
from PIL import Image
import tensorflow as tf

def process_image(image_path):
    """
    Args:
        - image_path
    Returns:
        - Preprocessed image
    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255
    return image
