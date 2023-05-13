'''
Script for Assignment 1, Visual Analytics, Cultural Data Science, F2023

The script is used to compare a target image to all other images in a corpus,
returning the N most similar images and their distance scores to the target image in a .csv file in the "out" directory (a plot is also outputted).

The search algorithm is either a simple search algorithm built on comparing color histograms or using KNN with a image features extracted with a pretrained CNN (VGG16).

In the terminal, run the script by typing: 
    python src/image_search.py -i {IMAGE} -N {N_IMAGES_TO_COMPARE_TO} -alg {ALGORITHM_TO_CHOSE}

Additional arguments for running the script
    -i   (for image, defaults to image_0020.jpg) 
    -n   (for N images to compare to, defaults to 5)
    -alg (for algorithm to use ('KNN' or 'SIMPLE'), defaults to 'SIMPLE')
'''

# utils 
import pathlib
import argparse

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-i", "--image", help = "chosen image (filename)", type = str, default = "image_0020.jpg") #default img defined
    parser.add_argument("-n", "--number", help = "number of images to compare to chosen image", type = int, default = 5) #default n images (5)
    parser.add_argument("-alg", "--algorithm", help = "image search algorithm. Can be either 'SIMPLE' or 'KNN'", type = str, default = "SIMPLE") #default n images (5)
    
    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main(): 
    # define args
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__) # define path to current file
    img_dir = path.parents[1] / "data" / "flowers" 
    out_dir = path.parents[1] / "out"

    # run either simple image search or KNN, depending on argument
    if args.algorithm == "SIMPLE":
        # import simple image search alg
        from simple_image_search_fns import simple_image_search
        
        # perform simple image search
        simple_image_search(args.image, img_dir, out_dir, args.number)

    elif args.algorithm == "KNN":
        # import VGG16, KNN image search alg
        from tensorflow.keras.applications.vgg16 import VGG16
        from knn_image_search_fns import KNN_image_search, extract_features_from_directory
        
        # initialize model (without classifier layers)
        model = VGG16(weights='imagenet', include_top=False, pooling='avg',input_shape=(224, 224, 3))

        # extract features
        features = extract_features_from_directory(img_dir, model)

        # perform KNN image search 
        KNN_image_search(args.image, img_dir, out_dir, features, args.number)

## run script ## 
if __name__ == "__main__":
    main()