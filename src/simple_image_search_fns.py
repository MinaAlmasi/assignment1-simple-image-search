'''
Script for Assignment 1, Visual Analytics, Cultural Data Science, F2023

The script comprises several functions which jointly make up a simple image search algorithm using the comparison of color histograms.

@MinaAlmasi
'''

# utils 
import pathlib 
from tqdm import tqdm 

# images, data wrangling 
import cv2
import pandas as pd

# plotting img search 
from plotting_fns import plot_image_search

## helper functions for image search algorithm ## 
def image_dir(dir:pathlib.Path()):
    '''
    Function which saves a list of all images ending with .jpg or .png in specified directory (dir). 

    Args
        - dir: directory where images are located

    Returns: 
        - images: list of all images in specified directory (dir)
    '''

    # only keep paths which are images
    images = [image.name for image in dir.iterdir() if image.name.endswith(".jpg") or image.name.endswith(".png")]

    return images


def image_hist_normalized(img, img_dir:str):
    '''
    Function which reads in an image and creates a normalized color histogram for the image.  

    Args: 
        - img: image filename
        - img_dir: directory where image is located

    Returns: 
        - norm_hist: normalized color histogram of image
    '''

    # read image
    image = cv2.imread(str(img_dir / img))

    # create hist 
    hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])

    # normalize hist
    norm_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

    return norm_hist   

## image search algorithm ## 
def simple_image_search(chosen_image:str, img_dir:pathlib.Path(), out_dir:pathlib.Path(), n:int, chosen_img_dir:pathlib.Path()=None):
    '''
    Function which compares a select image with other images in a specified directory. 
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
    
    # if no directory is specified for the chosen image, make same as img_dir 
    if chosen_img_dir is None:
        chosen_img_dir = img_dir

    # make sure path exists, otherwise make it
    out_dir.mkdir(parents=True, exist_ok=True)

    # list of all images
    images = image_dir(img_dir)

    # remove chosen image from list of image paths
    if chosen_image in images:
        images.remove(chosen_image)

    # chosen image 
    chosen_hist = image_hist_normalized(chosen_image, chosen_img_dir)
    chosen_data = pd.DataFrame({"Image":chosen_image, "Distance":[0], "Target (y/n)":["y"]})    

    # list over dataframes
    data_imgs = [chosen_data]

    # loop over images 
    for image in tqdm(images, total = len(images), desc="Comparing images"): #tqdm = show progress bar 
        # normalize hist 
        image_hist = image_hist_normalized(image, img_dir)

        # create data frame
        data = pd.DataFrame()

        # define image file name and whether it is target
        data["Image"] = [image]
        data["Target (y/n)"] = ["n"]
        
        # calculate distance score 
        data["Distance"] = [cv2.compareHist(chosen_hist, image_hist, cv2.HISTCMP_CHISQR)]

        # append dataframe to data list 
        data_imgs.append(data)

    # concatenate
    final_data = pd.concat(data_imgs, ignore_index = True)

    # sort data
    final_data = final_data.sort_values(by=["Distance"], ascending = True, ignore_index = True)

    # save only n+1 (chosen image and n most similar)
    final_data = final_data.head(n+1)

    # save dataframe to csv 
    final_data.to_csv(out_dir / f"SIMPLE_{n}_imgs_similar_to_{chosen_image}.csv")

    # plot data 
    plot_image_search(chosen_image, final_data, n, img_dir, out_dir, alg_name="SIMPLE")

    # print statement with rounded distance values for overview 
    print(f"\nPrinting the {n} most similar images to {chosen_image}: \n {final_data.iloc[1:, [0,1]].round(2)}\n") # only select similar imgs (exclude 1st row), select only first 2 cols