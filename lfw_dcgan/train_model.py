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


file_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=''' DCGAN model to train on Labeled Faces in the Wild (LFW) Dataset ''')

parser.add_argument('dataset_path', help='Path to dataset folder')
parser.add_argument('epoch_st', help='Epoch to start training')
args = parser.parse_args()

dataset_path = args.dataset_path

image_set = utils.celeb_face_loader_64(dataset_path)

if not os.path.exists(file_dir + "/gen_kernel_filters"):
    os.mkdir(file_dir + "/gen_kernel_filters")
if not os.path.exists(file_dir + "/disc_kernel_filters"):
    os.mkdir(file_dir + "/disc_kernel_filters")
if not os.path.exists(file_dir + "/Generated_per_epoch"):
    os.mkdir(file_dir + "/Generated_per_epoch")


real_X = T.ftensor4()
Z = T.fmatrix()


latent_size = 100
batch_size = 16
img_ht = 64
img_wdt = 64


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    fan_in = np.prod(shape[1:])
    fan_out = (shape[0] * np.prod(shape[2:]))

    local=np.random.randn(*shape)
    W_bound = np.sqrt(2.0/(fan_in + fan_out))

    return theano.shared(floatX(local*W_bound))

def init_latent_space():
    """
    Initialization of 100 dimensional latent space
    """
    #RandomState = np.random.RandomState(seed_value)

    space = np.random.uniform(0,1,(batch_size,latent_size))
    space = np.float32(space)
    return space

def generator_model(gX, gw1, gw2, gw3, gw4, gw5):

    l1 = relu(batchnorm(T.dot(gX,gw1)))
    l2 = relu(batchnorm(T.dot(l1, gw2)))
    l2a = l2.reshape((batch_size,128,8,8))
    l3 = relu(batchnorm(conv2d_transpose(l2a,gw3, input_shape=(batch_size,64,16,16), border_mode=(2,2), subsample=(2, 2))))
    l4 = relu(batchnorm(conv2d_transpose(l3,gw4, input_shape=(batch_size,64,32,32), border_mode=(2,2), subsample=(2, 2))))
    l5 = T.tanh((conv2d_transpose(l4, gw5, input_shape=(batch_size,3,64,64), border_mode=(2,2), subsample=(2,2))))

    return l5

def discriminator_model(dX, dw1, dw2, dw3, dw4, dw5):

    l1 = relu(batchnorm(conv2d(dX,dw1, subsample=(2, 2), border_mode=(2,2))))
    l2 = relu(batchnorm(conv2d(l1,dw2, subsample=(2, 2), border_mode=(2,2))))
    l3 = relu(batchnorm(conv2d(l2,dw3, subsample=(2, 2), border_mode=(2,2))))
    l3a = l3.flatten(2)
    l4 = relu(batchnorm(T.dot(l3a, dw4)))
    l5 = sigmoid((T.dot(l4, dw5)))

    return l5


'''layer size specs'''
gen_size = []
dis_size = []

dis_size.append((32,3,5,5))
dis_size.append((64,32,5,5))
dis_size.append((128,64,5,5))
dis_size.append((1*128*8*8,1024))
dis_size.append((1024,1))

gen_size.append((100,1024))
gen_size.append((1024,1*128*8*8))
gen_size.append((128,64,5,5))
gen_size.append((64,64,5,5))
gen_size.append((64,3,5,5))

params_disc = []
params_gen = []

for size in dis_size:
    cur_weight = init_weights(size)
    params_disc.append(cur_weight)

for size in gen_size:
    cur_weight = init_weights(size)
    params_gen.append(cur_weight)

gen_X = generator_model(Z, *params_gen) #generated image from latent vec

p_real = discriminator_model(real_X, *params_disc) #output class of real data from discriminator
p_gen = discriminator_model(gen_X, *params_disc) #output class of generated data from discriminator

d_cost_real = binary_crossentropy(p_real, T.ones(p_real.shape)).mean() #real cost log(1 - D(xi)) term for discriminator
d_cost_gen = binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean() #generated cost log(D(G(zi))) term for discriminator
g_cost_d = binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean() #generator update log(1 - D(G(zi)))

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

d_updates = Adam_updates(d_cost, params_disc)
g_updates = Adam_updates(g_cost, params_gen)

print("Compiling...")

_train_g = theano.function([real_X, Z], cost, updates=g_updates)
_train_d = theano.function([real_X, Z], cost, updates=d_updates)
_gen = theano.function([Z], gen_X)

print("Compile Completed")

n_epochs = 10
train_dis_skip = 1
train_gen_skip = 1
start = int(args.epoch_st) #argument to start training from epoch + 1

'''to set saved parameters'''

if(start>0):
    last_epoch = start-1

    for layer_no in range(len(params_gen)):
        print("Reading gen layer number %d"%layer_no)

        weight_layer = np.load(file_dir + "/gen_kernel_filters/weight_layer_%d_%d.npy"%(last_epoch,layer_no+1))
        params_gen[layer_no].set_value(weight_layer)

    for layer_no in range(len(params_disc)):
        print("Reading discrim layer number %d"%layer_no)

        weight_layer = np.load(file_dir + "/disc_kernel_filters/weight_layer_%d_%d.npy"%(last_epoch,layer_no+1))
        params_disc[layer_no].set_value(weight_layer)

'''training'''

f=open("Epoch_stats.txt","w",0)
f.write("Epoch:	    Dis_Correct	  Gen_Correct   Average Dis Loss \n")
for epoch in range(start,n_epochs):
    dis_correct = 0
    gen_correct = 0
    sum_cost_d = 0
    idx_store = 1
    main_cost = []
    for idx in range(image_set.shape[0]/batch_size):
        latent_vec = init_latent_space()
        input_batch = image_set[batch_size*idx:batch_size*(idx+1),:,:,:]
        if(idx%train_dis_skip == 0):
            cost_dis = _train_d(input_batch, latent_vec) #discriminator update
            main_cost = cost_dis
        if(idx%train_gen_skip == 0):
            cost_gen = _train_g(input_batch, latent_vec) #generator update
            main_cost = cost_gen
        if(main_cost[3] < main_cost[4]):
            gen_correct +=1
        elif(main_cost[3] > main_cost[4]):
            dis_correct +=1

        #ideally win-lose ratio should be 50% for both (will oscillate)
        percentage_gen = gen_correct*100.0/(gen_correct + dis_correct)
        percentage_dis = dis_correct*100.0/(gen_correct + dis_correct)
        print(epoch, idx, dis_correct, gen_correct, percentage_dis, percentage_gen, main_cost[3], main_cost[4], main_cost[1])
        sum_cost_d += main_cost[1]
        idx_store = idx


    print("Epoch Stats:")
    print(epoch,dis_correct, gen_correct, sum_cost_d/idx_store)
    f.write(str(epoch) + "            "+str(dis_correct) + "             "+ str(gen_correct) + "             " + str(sum_cost_d/idx_store))
    f.write('\n')

    '''Generated Samples '''
    random_space = init_latent_space()
    generate = _gen(random_space)
    generate = np.swapaxes(generate, 1, 3) #hack: swap axis in image space

    print(generate.shape)
    panel = np.zeros((generate.shape[1], batch_size*generate.shape[2], 3))
    for idx in range(batch_size):
        panel[:,img_ht*idx:img_ht*(idx+1),:] = generate[idx]

    panel = np.uint8(panel*127.5 + 127.5)

    image_dim = generate.shape[1]
    saver_panel = np.zeros((image_dim*4,image_dim*4,3), dtype = np.uint8)
    for i in range(0,16,4):
        current_image = panel[:,(i)*image_dim:(i+4)*image_dim,:]
        row = i/4
        saver_panel[row*image_dim:(row+1)*image_dim,:,:] = current_image

    cv2.imwrite(file_dir + "/Generated_per_epoch/generated_image_%d.png"%(epoch), saver_panel)

    '''Saving Model'''
    if(epoch%1 ==0):
        for layer_no in range(len(params_gen)):
            print("Writing gen layer number %d"%layer_no)

            weight_layer = params_gen[layer_no].get_value()
            np.save(file_dir + "/gen_kernel_filters/weight_layer_%d_%d.npy"%(epoch,layer_no+1), weight_layer)

        for layer_no in range(len(params_disc)):
            print("Writing disc layer number %d"%layer_no)

            weight_layer = params_disc[layer_no].get_value()
            np.save(file_dir + "/disc_kernel_filters/weight_layer_%d_%d.npy"%(epoch,layer_no+1), weight_layer)


f.close()
