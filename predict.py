import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from PIL import Image
import json
import tensorflow_hub as hub
import argparse

image_size = 224
def process_image(image):
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(image_path, model, top_k, category_names):
    rmodel = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer})
    top_k = int(top_k)
    with open(category_names, 'r') as f:
        class_names = json.load(f)
### after looking at the flowers, by observation the wrong flower labels were assigned, so looking at the JSON file was necessary
### by looking at the JSON file, it can be seen that the labels are from 1 to 102. Since the indices are from 0 to 101, remapping
### is needed to obtain the correct flower names.
    class_names_remapped = dict()
    for i in class_names:
        class_names_remapped[str(int(i)-1)] = class_names[i]
    image=Image.open(image_path)
    converted_image= np.asarray(image)
    processed_converted_image= process_image(converted_image)
    input_image= np.expand_dims(processed_converted_image, axis=0)
    pred= rmodel.predict(input_image)
    top_kpred, top_kclass =tf.math.top_k(pred, k=top_k, sorted=True, name=None)
    names = [class_names_remapped[str(i)] for i in top_kclass.numpy()[0]]
    print(top_kpred.numpy()[0])
    print(top_kclass.numpy()[0])
    print(names)
    return top_kpred.numpy()[0], names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flower image classification')
    parser.add_argument('image_path', default="")
    parser.add_argument('model', default="")
    parser.add_argument('--top_k', default= 3)
    parser.add_argument('--category_names', default="label_map.json")
    args=parser.parse_args()
    predict(args.image_path,args.model,args.top_k,args.category_names)
