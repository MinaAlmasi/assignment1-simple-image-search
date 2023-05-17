'''
Script for Assignment 1, Visual Analytics, Cultural Data Science, F2023

The script comprises several functions which jointly make up an image search algorithm using KNN and pretrained VGG16. 

@MinaAlmasi
'''
# utils
import pathlib
from tqdm import tqdm

# tensorflow VGG16 for extracting features
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# utils for extracting features
import numpy as np
from numpy.linalg import norm

# KNN search algorithm
from sklearn.neighbors import NearestNeighbors

# data wrangling
import pandas as pd 

# for plotting images
from plotting_fns import plot_image_search

def extract_features(img_path:pathlib.Path(), model, verbose:int=1): # function adapted from class notebook
    """
    Extract features from image data using pretrained model (e.g., VGG16)

    Args:
        - img_path: path where the image is located
        - model: intialized pretrained model (e.g., VGG16)
        - verbose: whether the function should print information. Defaults to 1. (0 = No information).

    Returns: 
        - flattened, normalized features 
    """
    
    # Define input image shape
    input_shape = (224, 224, 3)
    
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    
    # preprocess image - see last week's notebook
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img, verbose=verbose) 
    
    # flatten featurres 
    flattened_features = features.flatten()
    
    # normalise features
    normalized_features = flattened_features / norm(features)

    return flattened_features

def extract_features_from_directory(img_dir:pathlib.Path(), model):
    '''
    Extract features from image data directory using a pretrained model (e.g., VGG16) 

    Args: 
        - img_dir: directory where the image data is located
        - model: intialized pretrained model (e.g., VGG16)

    Returns: 
        - features: dictionary with image filenames (keys) and flattened features for each image (values)
        - filepaths: dictionary with image filenames (keys) and filepaths (values)
    '''

    # empty dict with filenames, filepaths
    filepaths = {}

    # extract filenames by iterating over image directory (img_dir), but only if the file is a regular file (with if statement)
    for file in sorted(img_dir.iterdir()):
        if file.is_file(): 
            # define filepath with image directory and file name 
            filepath = img_dir / file.name  

            # append to filepaths dict  
            filepaths[file.name] = filepath

    # empty dict which will be filenames, features
    features_dict = {} 

    # iterate over filenames and filepaths in filepaths dictionary
    with tqdm(total=len(filepaths)) as pbar:
        for filename, filepath in filepaths.items():
            # extract feature 
            img_feature = extract_features(filepath, model, verbose=0) # verbose = 0 to make it more smooth when processing multiple images 

            # add filename and feature to features dictionary
            features_dict[filename] = img_feature

            # update progress bar
            pbar.update(1)

    return features_dict

def KNN_image_search(chosen_image, img_dir, out_dir, features_dict, n:int):
    '''
    Compare a select image with other images in a specified directory with KNN. 
    The function returns an overview of the N images with lowest distance (most similar images) as both CSV and plot.

    Args: 
        - chosen_image: image that every other image will be compared to 
        - img_dir: directory where images to be compared are located. If chosen_img_dir is not specified, img_dir will also be the directory of the chosen_image
        - out_dir: directory where the output should be saved 
        - n: number of images with smallest distances (most similar images)
        - chosen_img_dir: directory where chosen_image is located. Defaults to None. If specified as None, it will be the equal to img_dir

    Outputs: 
        - .csv file: file containing overview of top N most similar images (smallest distance to chosen image) in out_dir
        - .png image: plotted overview of all images and their distances as .PNG in out_dir. 
    '''
    # make lists from dictionary
    image_names = list(features_dict.keys())
    features = list(features_dict.values())

    # fit KNN 
    neighbours = NearestNeighbors(n_neighbors=n+1, 
                             algorithm='brute',
                             metric='cosine').fit(features) # fitted on all the values (features) in the features_dit

    # extract target features 
    target_features = features_dict[chosen_image]

    # calculate nearest neighbours for target
    distances, indices = neighbours.kneighbors([target_features])
    
    # add chosen data 
    chosen_data = pd.DataFrame({"Image":chosen_image, "Distance":[0], "Target (y/n)":["y"]})    

    # create dataframe 
    data_imgs = [chosen_data]

    # loop over nearest neighbors add to 
    for i in range(1, n+1):
        data = pd.DataFrame()
        data["Image"] = [image_names[indices[0][i]]] # take only keys of features_dict (image names), index based on the indices given by neighbours
        data["Distance"] = [distances[0][i]]
        data["Target (y/n)"] = ["n"]
        data_imgs.append(data)
    
    # concatenate 
    final_data = pd.concat(data_imgs, ignore_index = True)

    # save data
    final_data.to_csv(out_dir / f"KNN_{n}_imgs_similar_to_{chosen_image}.csv")

    # plot results
    plot_image_search(chosen_image, final_data, n, img_dir, out_dir, alg_name="KNN")
    
    # print statement with rounded distance values for overview
    print(f"\nPrinting the {n} most similar images to {chosen_image}: \n {final_data.iloc[1:, [0,1]].round(2)}\n") # only select similar imgs (exclude 1st row), select only first 2 cols