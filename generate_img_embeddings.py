# Image Embedding (Image Feature Vector) with PreTrained Models
# Visit TensorFlow Hub PreTrained Models
# https://tfhub.dev/s?module-type=image-feature-vector

import glob
import os
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tqdm import tqdm

from src.utils import get_project_config

tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings(action="ignore")

base_list = []
embed_dict_input = {}
flattended_feature_list = []

def extract_embeedding(file, grayscale_mode=False):
    """
        Embedding Extraction Function
    """
    global base_list
    global embed_dict_input
    global flattended_feature_list
    
    # GrayScale Mode
    if grayscale_mode:
        # https://stackoverflow.com/questions/52307290/what-is-the-difference-between-images-in-p-and-l-mode-in-pil
        # https://www.geeksforgeeks.org/python-pil-image-convert-method/
        file_ = Image.open(file).convert('L').resize(IMAGE_SHAPE)
        file_ = np.stack((file_,)*3, axis=-1)
        file_ = np.array(file_)/255.0
    else:
        file_ = Image.open(file).resize(IMAGE_SHAPE)
        file_ = np.array(file_)/255.0

    embedding = model.predict(file_[np.newaxis, ...], verbose=0)
    feature_vector = np.array(embedding)
    flattended_feature = feature_vector.flatten()

    base_name = os.path.basename(file)
    embed_dict_input[base_name] = flattended_feature

    base_list.append(base_name)
    flattended_feature_list.append(flattended_feature)

def main(input_dir, results_name):
    """
        Main Function
    """
    input_data_list = glob.glob(os.path.join(input_dir, '*'))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(extract_embeedding, input_data_list), total=len(input_data_list)))

    # with open("data/result/" + results_name +'.pickle', 'wb') as handle:
    #     pickle.dump(embed_dict_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_emb_data = pd.DataFrame.from_dict(embed_dict_input, orient='index').reset_index()
    df_emb_data.rename(columns={'index':'ImgName'}, inplace=True)
    df_emb_data.to_csv(results_name + ".csv", index=False)

if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, 'data')
    PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
    IMAGE_DATA_PATH = os.path.join(DATA_PATH, 'images')
    RESULT_DATA_PATH = os.path.join(DATA_PATH, 'result')

    pre_trained_model_dict = get_project_config(cfg_file='pre_trained_model.json')

    for idx, mdl_name in enumerate(pre_trained_model_dict.keys()):
        print("Model Name:", mdl_name)
        MODEL_URL = pre_trained_model_dict.get(mdl_name)['MODEL_URL']

        # IMAGE_SHAPE = tuple(pre_trained_model_list[mdl_name]['IMAGE_SHAPE'])
        IMAGE_SHAPE = (224, 224)
        # print(MODEL_URL, IMAGE_SHAPE, type(tuple(IMAGE_SHAPE)))

        layer = hub.KerasLayer(MODEL_URL)
        model = tf.keras.Sequential([layer])

        img_dir = IMAGE_DATA_PATH
        result_dir = PROCESSED_DATA_PATH
        result_name = os.path.join(result_dir, mdl_name)
        tic = time.time()
        main(input_dir=img_dir, results_name=result_name)
        toc = time.time()
        print("Elapsed Time(Min):", (toc - tic) / 60)
        print("- " * 25)