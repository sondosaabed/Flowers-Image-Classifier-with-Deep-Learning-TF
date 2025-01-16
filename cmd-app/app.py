import json
from scripts.predict import predict
from scripts.parser import get_args
import tensorflow as tf
import tensorflow_hub as hub

args = get_args()

image_path = args.image_path
saved_keras_model = args.saved_keras_model
top_k = args.top_k
category_names = args.category_names

print('Path to the input image:', image_path) 
print('Path to the keras model:', saved_keras_model)
print('Top k class probabilities:',top_k) 
print('Path to the JSON file:', category_names)


## Load a JSON file that maps the class values to other category names
with open(category_names, 'r') as f:
    class_names = json.load(f)


## Load the saved model 
loaded_keras_model = tf.keras.models.load_model(saved_keras_model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

print(loaded_keras_model.summary())

## Return the top K classes along with associated probabilities
top_k_probs, top_k_classes, _ = predict(image_path, loaded_keras_model, top_k)

print('List of flower labels along with corresponding probabilities:', top_k_classes, top_k_probs)

## print unknown too
for i in range(len(top_k_classes)): 
    print(f'Flower Name: {class_names.get(str(top_k_classes[i] + 1), "Unknown")}')
    print(f'Class Probability: {top_k_probs[i]:.4f}')