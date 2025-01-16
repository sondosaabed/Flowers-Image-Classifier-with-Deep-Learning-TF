import argparse


# Creat a parser
parser = argparse.ArgumentParser(description="A cmd application helps to identify type of flower using the image.")

# Positional arguments 
parser.add_argument("image_path", help="path to the input image folder", type=str)
parser.add_argument("saved_keras_model", help="path to the saved Keras model", type=str)

# Optional arguments 
parser.add_argument("-k", "--top_k", default=3, help ="top k class probabilities", type=int)
parser.add_argument("-n", "--category_names", default="./label_map.json", help="path to a JSON file mapping labels to the actual flower names", type=str)

