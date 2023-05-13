# Simple Image Search Algorithm and KNN

This repository forms *assignment 1* by Mina Almasi (202005465) in the subject *Visual Analytics*, *Cultural Data Science*, F2023. The assignment description can be found [here](https://github.com/MinaAlmasi/assignment1-simple-image-search/blob/master/assignment-desc.md).

The repository contains code for building and running a simple image search algorithm that relies on colour histograms to compare similarities within images. As a bonus, a KNN image search algorithm is also constructed which relies on a pretrained CNN (VGG16) to extract features from image data. 

Concretely, both algorithms involve choosing a target image and comparing it to other images, returning the ```N``` most similar images and their distances as a ```.CSV``` file. As an addition, the script also returns a plot of the images with the target image highlighted in red (see [Results](https://github.com/MinaAlmasi/assignment1-simple-image-search/tree/master#results)).

## Data 
The image searches are performed on the [*Flowers*](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) dataset. The dataset contains 1360 images of 17 types of flowers that are common in the UK (i.e., 80 images per type).

## Reproduciblity
To reproduce the results, please follow the instructions in the [*Pipeline*](https://github.com/MinaAlmasi/assignment1-simple-image-search/tree/master#pipeline) section. 

## Project Structure 
The repository is structured as such:
```
├── README.md
├── assignment-desc.md
├── data
│   └── flowers.zip                                <---     unzip prior to search
├── out                                            <---    .CSV & .png from search
│   ├── KNN_5_imgs_similar_to_image_0020.jpg.csv 
│   ├── KNN_5_imgs_similar_to_image_0020.jpg.png
│   ├── SIMPLE_5_imgs_similar_to_image_0020.jpg.csv
│   └── SIMPLE_5_imgs_similar_to_image_0020.jpg.png
├── requirements.txt
├── run.sh                                          <---   run to perform image searches 
├── setup.sh                                        <---   run to install necessary reqs & packages
└── src
    ├── image_search.py                             <---   run either KNN or SIMPLE
    ├── knn_image_search_fns.py                     <---   functions for KNN search
    ├── plotting_fns.py                             <---   functions for plotting
    └── simple_image_search_fns.py                  <---   functions for SIMPLE search
```

The ```.CSV``` file for the chosen image is structured as such for both algorithms:
|Image|Distance|Target (y/n)
|---|---|---|
|image_XXXX.jpeg |0.0|y|
|image_YYYY.jpeg|---|n|
|image_ZZZZ.jpeg|---|n|

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). 
Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Setup
Firstly, please unzip the ```flowers.zip``` in the **data** folder. Then, run ```setup.sh``` in the terminal:
```
bash setup.sh
```
The script installs the necessary packages and its dependencies in a newly created virtual environment (```env```). 

### Running the Image Search Algorithms
To run the image searches, type the following in the terminal:
```
bash run.sh
```

This will run **both** image search algorithms for the default image (*image_0020.jpeg*) and output the five most similar images.

### Custom Image Search
You can run the python script with custom arguments as such:
```
python src/image_search.py -i {IMAGE} -N {N_IMAGES_TO_COMPARE_TO} -alg {ALGORITHM_TO_CHOSE}
```

| Arg        | Description                                         | Default         |
| :---       |:---                                                 |:---             |
| ```-i```   | chosen image to compare with                        | image_0020.jpg  |
| ```-n```   | N images to compare chosen image with               | 5               |
| ```-alg``` | image search algorithm, choose either SIMPLE or KNN | SIMPLE          |

## Results 
The images below show the results of running the two search algorithms on ```image_0020.jpg```.

### Simple Image Search Algorithm 
<p align="left">
  <img width=65% height=65% src="https://github.com/MinaAlmasi/assignment1-simple-image-search/blob/master/out/SIMPLE_5_imgs_similar_to_image_0020.jpg.png">
</p>

### KNN Image Search Algorithm
<p align="left">
  <img width=65% height=65% src="https://github.com/MinaAlmasi/assignment1-simple-image-search/blob/master/out/KNN_5_imgs_similar_to_image_0020.jpg.png">
</p>

### Discussion of Results
For this particular example, the ```KNN algorithm``` is undeniably superior to the ```simple image search```. The simple image search algorithm has three yellow flowers that do not seem to be the same type along with both a white and purple flower. On the other hand, the KNN algorithm contains only yellow images and several images appear to also be the same flower type as ```image_0020.jpg```. 

However, it is difficult to conclude anything without doing any exhaustive testing as it may also be an image that is particularly easy for the KNN algorithm, making it seem like it is much more superior than it actually is. 

## Author 
This repository was created by Mina Almasi:
- github user: @MinaAlmasi
- student no: 202005465, AUID: au675000
- mail: mina.almasi@post.au.dk 