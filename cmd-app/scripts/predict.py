from scripts.utils import process_image
import numpy as np
import tensorflow as tf

def predict(image_path, model, top_k):
    """
    Args:
        - image_path
        - model
        - top_k

    Returns:
        returns the top $K$ most likely class labels along with the probabilities.
    """
    processed_image = process_image(image_path)
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    model_prediction = model.predict(expanded_image)
     
    top_k_probs, top_k_classes = tf.nn.top_k(model_prediction, k=top_k)
    top_k_probs = list(top_k_probs.numpy()[0])
    top_k_classes = list(top_k_classes.numpy()[0])
     
    return top_k_probs, top_k_classes, processed_image 