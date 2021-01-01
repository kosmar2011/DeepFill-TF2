Tensorflow 2.x implementation of "Free-Form Image Inpainting with Gated Convolution" (https://github.com/JiahuiYu/generative_inpainting). 
Sketch generation for user guidance missing.

GeneratorMulticolumn() has the same coarse-fine structure as "Free-Form Image Inpainting with Gated Convolution", but with a multi-column coarse stage inspired by "Image Inpainting via Generative Multi-column Convolutional Neural Networks".

![alt text](https://github.com/kosmar2011/ImageInpaintingGatedConv-TF2/blob/master/table.PNG?raw=true)

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
