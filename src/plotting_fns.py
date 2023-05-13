'''
Script for Assignment 1, Visual Analytics, Cultural Data Science, F2023

This script contains functions that serve to plot the image search algorithms' results. 
The plotting relies on the .CSV files that contain the information such as image filenames and distances to target image.

@MinaAlmasi
'''

# utils
import pathlib

# plotting 
import cv2
import matplotlib.pyplot as plt
from math import ceil

def color_convert(img, img_dir):
    '''
    Function which converts cv2's BGR scale to RGB to plot images with matplotlib.

    Args: 
        -img: image to be color converted (filename)
        -img_dir: directory where image is located

    returns: 
        - rgb_img: image that is color converted 
    '''

    # read image
    img = cv2.imread(str(img_dir / img))

    # split image into seperate color channels
    b, g, r = cv2.split(img)

    # merge color channels into rgb order 
    rgb_img = cv2.merge([r,g,b])

    return rgb_img


def plot_image_search(chosen_image, data, n:int, img_dir:str, out_dir:str, alg_name:str, chosen_img_dir=None):
    '''
    Function which plots a chosen image (highlighted in red) and its N most similar images. 

    Args: 
        - chosen_image: chosen_image which has been compared to the similar images
        - data: Pandas DataFrame containing all similar images and their distance scores to chosen_image
        - n: number of images with smallest distances to chosen_image (i.e., most similar images)
        - alg_name: whether the algorithm is SIMPLE or KNN
        - chosen_img_dir: directory where chosen_image is located. Defaults to None. If specified as None, it will be the equal to img_dir

    Outputs: 
        - .png image: plotted overview of all images and their distances as .PNG in out_dir

    '''
    # if no directory is specified for the chosen image, make same as img_dir 
    if chosen_img_dir is None:
        chosen_img_dir = img_dir

    # create 2 rows of subplots, vary cols depending on amount of images. Always round up w/ ceil() in case of odd number of imgs
    fig, ax = plt.subplots(nrows=2, ncols=ceil(len(data)/2))

    # loop over subplots
    for i, ax in enumerate(ax.flatten()):
        # special plot for chosen_image (highlighted in red)
        if i == 0: 
            # read in image, convert to matplotlib RGB
            img = color_convert(chosen_image, img_dir)
            
            # set title for subplot
            ax.set_title("TARGET IMAGE \n" + str(data["Image"][i]), fontsize = 8, color="crimson") 
            
            # show image
            ax.imshow(img)
            
            # create red border around chosen image, remove ax ticks
            ax.patch.set_edgecolor("crimson") 
            ax.patch.set_linewidth(5)  
            ax.tick_params(axis="both", which="both", bottom=False, top=False, left= False, labelbottom=False, labelleft=False) 
        else: 
            try: 
                # plot all other images
                img = color_convert(str(data["Image"][i]), img_dir)
                ax.imshow(img) 
                ax.set_title(str(data["Image"][i])+ "\nDistance: " + str(data["Distance"][i].round(2)), fontsize = 7) 
                ax.axis("off") # remove axis completely 
            except KeyError: # if keyerror is raised (due to no more similar images), make axis white (as there will be an extra subplot, but no image)
                ax.axis("off")
    
    # ensure no overlapping images
    fig.tight_layout()

    # save plot
    plt.savefig(out_dir / f"{alg_name}_{n}_imgs_similar_to_{chosen_image}.png", dpi=300)