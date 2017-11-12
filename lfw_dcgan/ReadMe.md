### LFW_DCGAN

Implementation of [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) trained on the [LFW](http://vis-www.cs.umass.edu/lfw/) dataset in Theano.

Trained and Tested on the Funneled dataset. 

![Training Phase](https://media.giphy.com/media/l2QDPLUSp3Im2OVwc/giphy.gif)

Download dataset: https://drive.google.com/open?id=0Bwt2AQJsHr2wV1d2YUZLOGp5Q1U

To train from scratch: 
    
    $ python train_model.py /path/to/Dataset 0

to restart training from a particular epoch:

    $ python train_model.py /path/to/Dataset (epoch_no+1)

to Generate samples from random vec
    
    $ python Generate.py epoch_no

