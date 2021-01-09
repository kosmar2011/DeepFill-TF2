Tensorflow 2.x implementation of ["Free-Form Image Inpainting with Gated Convolution"](https://github.com/JiahuiYu/generative_inpainting). 
This repository does not include everything. For example it is missing user guidance. 

You don't need the neural gym library dependencies to run. 

It also includes an alternative generator architecture with the name GeneratorMulticolumn(). It has the same coarse-fine structure as "Free-Form Image Inpainting with Gated Convolution", but with a multi-column coarse stage inspired by "Image Inpainting via Generative Multi-column Convolutional Neural Networks".


DIRECTORIES

    ├── ImageInpaintingGatedConv-TF2

        ├── config.py

        ├── test_epoch.py

        ├── training.py

        ├── utils.py

        ├── sn.py

        └── net.py

    ├── training_checkpoints

    ├── inpaint.yml

    ├── TRAIN #Contains training images 
    
    └── TEST  #Contains inference/testing images
