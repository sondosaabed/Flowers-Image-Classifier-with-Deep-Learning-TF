import json
import tensorflow as tf
import tensorflow_hub as hub
import predict
import parser

args = parser.parse_args()   

image_path = args.image_path
saved_keras_model = args.saved_keras_model
top_k = args.top_k
category_names = args.category_names

print('Path to the input test image:', image_path) 
print('Path to the saved keras model:', saved_keras_model)
print('top k class probabilities:',top_k) 
print('Path to the json file:', category_names)


# Load a JSON file that maps the class values to other category names
with open(category_names, 'r') as f:
    class_names = json.load(f)


# Load the saved Keras model 
loaded_keras_model = tf.keras.models.load_model(saved_keras_model, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

print(loaded_keras_model.summary())

# Call the function that retures the top K classes along with associated probabilities
top_k_probs, top_k_classes = predict(image_path, loaded_keras_model, top_k) 
print('List of flower labels along with corresponding probabilities:', top_k_classes, top_k_probs)

for flower in range(len(top_k_classes)): 
    print('Flower Name:', class_names.get(str(top_k_classes[flower]+1)))
    print('Class Probabilty:', top_k_probs[flower])