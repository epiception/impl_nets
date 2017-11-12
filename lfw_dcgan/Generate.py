import os
import argparse
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs as conv2d_transpose
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

from theano.tensor.nnet.nnet import softmax, relu, categorical_crossentropy, binary_crossentropy, sigmoid
from theano.tensor.nnet.bn import batch_normalization as bn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.tensor.signal.pool import pool_2d
from tqdm import tqdm
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import utils
from training_utils import *

Z = T.fmatrix()

latent_size = 100
batch_size = 16
img_ht = 64
img_wdt = 64

file_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description=''' DCGAN Generator from Random Latent Vectors ''')

parser.add_argument('load_epoch', help='Epoch to load weights')
args = parser.parse_args()
load_epoch = int(args.load_epoch)

if not os.path.exists(file_dir + "/Generated_examples"):
    os.mkdir(file_dir + "/Generated_examples")

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_latent_space():
    """
    Initialization of 100 dimensional latent space
    """
    #RandomState = np.random.RandomState(seed_value)

    space = np.random.uniform(0,1,(batch_size,latent_size))
    #space = np.reshape(space,(latent_size,1))
    space = np.float32(space)
    return space

def init_weights(shape):
    fan_in = np.prod(shape[1:])
    fan_out = (shape[0] * np.prod(shape[2:]))

    local=np.random.randn(*shape)
    W_bound = np.sqrt(2.0/(fan_in + fan_out))

    return theano.shared(floatX(local*W_bound))

def generator_model(gX, gw1, gw2, gw3, gw4, gw5):

    l1 = relu(batchnorm(T.dot(gX,gw1)))
    l2 = relu(batchnorm(T.dot(l1, gw2)))
    l2a = l2.reshape((batch_size,128,8,8))
    l3 = relu(batchnorm(conv2d_transpose(l2a,gw3, input_shape=(batch_size,64,16,16), border_mode=(2,2), subsample=(2, 2))))
    l4 = relu(batchnorm(conv2d_transpose(l3,gw4, input_shape=(batch_size,64,32,32), border_mode=(2,2), subsample=(2, 2))))
    l5 = T.tanh((conv2d_transpose(l4, gw5, input_shape=(batch_size,3,64,64), border_mode=(2,2), subsample=(2,2))))

    return l3,l4,l5

gen_size = []
gen_size.append((100,1024))
gen_size.append((1024,1*128*8*8))
gen_size.append((128,64,5,5))
gen_size.append((64,64,5,5))
gen_size.append((64,3,5,5))

params_gen = []

for size in gen_size:
    cur_weight = init_weights(size)
    params_gen.append(cur_weight)

gen3, gen4, gen_X = generator_model(Z, *params_gen)

print("Compiling Generator Function")
_gen = theano.function([Z], outputs=[gen3, gen4, gen_X])
print("Compiled")



for layer_no in range(len(params_gen)):
    print("Reading gen layer number %d"%layer_no)

    weight_layer = np.loadtxt(file_dir + "/gen_kernel_filters/weight_layer_%d_%d.txt"%(load_epoch,layer_no+1), dtype = np.float32)
    weight_layer = weight_layer.reshape(gen_size[layer_no])
    params_gen[layer_no].set_value(weight_layer)

'''Generated Samples '''
for i in range(25):
    random_space = init_latent_space()
    output = _gen(random_space)
    ac3 = output[0];ac4 = output[1]; generate = output[2]
    generate = np.swapaxes(generate, 1, 3) #hack but works

    activation_3_panel = np.zeros((batch_size*ac3.shape[2], ac3.shape[1]*ac3.shape[3]))
    activation_4_panel = np.zeros((batch_size*ac4.shape[2], ac4.shape[1]*ac4.shape[3]))
    for current in range(batch_size):
        ac_temp_3 = ac3[current]
        ac_temp_4 = ac4[current]
        ac_temp_3 = np.reshape(ac_temp_3,(ac_temp_3.shape[0]*ac_temp_3.shape[1],ac_temp_3.shape[2])).T
        ac_temp_4 = np.reshape(ac_temp_4,(ac_temp_4.shape[0]*ac_temp_4.shape[1],ac_temp_4.shape[2])).T
        activation_3_panel[current*ac_temp_3.shape[0]:(current+1)*ac_temp_3.shape[0],:] = ac_temp_3
        activation_4_panel[current*ac_temp_4.shape[0]:(current+1)*ac_temp_4.shape[0],:] = ac_temp_4

    panel = np.zeros((generate.shape[1], batch_size*generate.shape[2], 3))
    for idx in range(batch_size):
        panel[:,img_ht*idx:img_ht*(idx+1),:] = generate[idx]
    panel = np.uint8(panel*127.5 + 127.5)
    cv2.imshow("Window1", activation_3_panel)
    cv2.imshow("Window2", activation_4_panel)
    cv2.imshow("output", panel)
    cv2.imwrite(file_dir + "/Generated_examples/generated_%d.png"%i, panel)
    cv2.waitKey(100)
