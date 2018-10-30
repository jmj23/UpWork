
# coding: utf-8

# Import libraries
import os
import sys
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, Dense, Reshape
from keras.layers import LeakyReLU, BatchNormalization, UpSampling2D
from keras.layers import concatenate, GlobalAveragePooling2D
from keras.models import Model
from keras.initializers import RandomNormal, glorot_uniform
from keras.datasets import cifar10
from keras.optimizers import SGD
import keras.backend as K
from tqdm import tqdm, trange
from scipy.signal import medfilt
from matplotlib import pyplot as plt


# Load CIFAR training 
# If not already downloaded, it will be downloaded first
# before loading
(cifar_images, _), (_, _) = cifar10.load_data()
# Normalize data to [0,1]
cifar_images = (cifar_images)/255



# Weight initializers
# initializer for generator
init1 = RandomNormal(0, 0.01)
# initializer for discriminator
init2 = glorot_uniform()


#%% Create GAN generator model

# input layer that accepts our 1D input noise vectors
g_input = Input(shape=(2048,),name='NoiseInput')
# reshape for use in convolutional layers
x = Reshape((4,4,128))(g_input)
# first deconvolutional layer with batchnorm and leaky relu
x = Conv2DTranspose(256,(3,3),padding='same',kernel_initializer=init1)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
# set of 4 convolutional layers
for _ in range(4):
    x = Conv2D(256,(5,5),padding='same',kernel_initializer=init1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
# Bilinear upsampling
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
# Set of 5 convolutional layers
for _ in range(5):
    x = Conv2D(256,(5,5),padding='same',kernel_initializer=init1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
# Bilinear upsampling
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
# Set of 5 convolutional layers
for _ in range(5):
    x = Conv2D(256,(5,5),padding='same',kernel_initializer=init1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
# Bilinear upsampling
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
# Set of 5 convolutional layers
for _ in range(5):
    x = Conv2D(256,(5,5),padding='same',kernel_initializer=init1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
# final convolutional layer
x = Conv2D(3,(5,5),padding='same',kernel_initializer=init1,activation='tanh')(x)

# Put model together
GenModel = Model(g_input,x)


# Display model summary to demonstrate correct network architecture
GenModel.summary()


# Create GAN discriminator model using an InceptionV3 format
# base number of filters to use
filtnum = 32
# Discriminator input
d_input = Input(shape=(32,32,3),name='input')
# inception block 1
rr = 1
x1 = Conv2D(filtnum*(2**(rr-1)), (1, 1),padding='same',kernel_initializer=init2)(d_input)
x1 = LeakyReLU(alpha=.1)(x1)
x3 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=init2)(d_input)
x3 = LeakyReLU(alpha=.1)(x3)
x51 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=init2)(d_input)
x51 = LeakyReLU(alpha=.1)(x51)
x52 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=init2)(x51)
x52 = LeakyReLU(alpha=.1)(x52)
x = concatenate([x1,x3,x52])
x = Conv2D(filtnum*(2**(rr-1)),(1,1),padding='valid',kernel_initializer=init2)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Conv2D(filtnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),kernel_initializer=init2)(x)
x = LeakyReLU(alpha=0.2)(x)


# repeated inception blocks
for rr in range(2,4):
    x1 = Conv2D(filtnum*(2**(rr-1)), (1, 1),padding='same',kernel_initializer=init2)(x)
    x1 = LeakyReLU(alpha=.1)(x1)
    x3 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=init2)(x)
    x3 = LeakyReLU(alpha=.1)(x3)
    x51 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=init2)(x)
    x51 = LeakyReLU(alpha=.1)(x51)
    x52 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=init2)(x51)
    x52 = LeakyReLU(alpha=.1)(x52)
    x = concatenate([x1,x3,x52])
    x = Conv2D(filtnum*(2**(rr-1)),(1,1),padding='valid',kernel_initializer=init2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filtnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),kernel_initializer=init2)(x)
    x = LeakyReLU(alpha=0.2)(x)   

# Use global average pooling to produce a single output
# which is the discriminator score
x = GlobalAveragePooling2D()(x)
# Put model together
DisModel = Model(d_input,x)


# Display discriminator model summary
DisModel.summary()


# Create training functions
lrD = 1e-4  # discriminator learning rate
lrG = 1e-4  # generator learning rate

#%% Setup training graph
# In this section, various graphs will be constructed
# so as to properly connect the generator model to the 
# discriminator model for training.
# The models will be trained using an improved
# Wasserstein GAN loss which includes a gradient penalty,
# Source: https://arxiv.org/abs/1704.00028
# This penalty and the total loss function will be calculated
# and conformed into a training function so that 
# tensorflow can calculate gradients and train
# each model.

# First, grab input and output tensors
noise_input = GenModel.inputs[0]
fake_output = GenModel.outputs[0]
real_output = DisModel.inputs[0]

# Get the output scores from the discriminator
# for each type of input:
# - real images
# - fake images generated by generator model
# These come as scalar "scores". We wish to train
# the discriminator model to predict the highest
# score for real images and the lowest score for
# fake images. This allows it to provide feedback
# to the generator model without vanishing gradients
# that occur in traditional GAN losses.
# real image score
realImScore = DisModel([real_output])
# fake image score
fakeImScore = DisModel([fake_output])
# create mixed output for gradient penalty
# Details are not important. This creates
# a mixed input image and gets the discriminator
# score on it for use in the gradient penalty
# that is necessary for the Wasserstein loss.
ep_input = K.placeholder(shape=(None,1,1,1))
mixed_output = Input(shape=(32,32,3),
                    tensor=ep_input * real_output + (1-ep_input) * fake_output)
mixed_score = DisModel([mixed_output])
# Average the scores output by the 
# discriminator to get the real and fake losses
realDloss = K.mean(realImScore)
fakeDloss = K.mean(fakeImScore)
# Calculate gradient penalty loss
grad_mixed = K.gradients([mixed_score],[mixed_output])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# The composite Discriminator loss combines
# the gradient penalty, and both real and fake
# losses so that it learns from all of them
loss_D = fakeDloss - realDloss + 1 * grad_penalty

#%% Next we define the loss for the image generator
# This is simply the opposite of the fake loss
# from the discriminator, since the generator wants to
# make that loss as high as possible to convince the
# discriminator that the images are real
loss_G = -fakeDloss

#%% Define training functions
# First, discriminator training function
# These lines make a tensorflow function for updating
# discriminator weights using Stochastic Gradient Descent
# based on the loss function that we defined
D_trups = SGD(lr=lrD,momentum=0.9,nesterov=True).get_updates(DisModel.trainable_weights,[],loss_D)
fn_trainD = K.function([noise_input, real_output, ep_input],[loss_D], D_trups)

# Generator Training function
G_trups = SGD(lr=lrG,momentum=0.9,nesterov=True).get_updates(GenModel.trainable_weights,[],loss_G)
fn_trainG = K.function([noise_input], [loss_G], G_trups)

# Now we finish setting up training
# Set number of iterations to do
numIter = 10000
# set batch size
b_s = 64
# preallocate for the training and validation losses
# so we can plot them after training
dis_loss = np.zeros((numIter,1))
gen_loss = np.zeros((numIter,1))

# Now the models will be trained simultaneously
# in a loop using the functions that were defined
# to update the weights.
# Batches of noise vectors and real images
# will be fed in to the discriminator so it learns
# to discriminate between real images and ones
# generated by the generator
# Noise batches will be fed into the generator
# so it learns to make convincing looking images
# Delete progress bar variable and reset

# Reset progress bar first
if 't' in locals():
    t.close()
    del t
# Create new progress bar
t = trange(numIter,file=sys.stdout)
# Main training loop
for ii in t:
    # Train discriminator multiple times
    # for each generator batch. In this case,
    # a ratio of 5
    for _ in range(5):
        # Train Discriminator in this nested loop
        # Grab random training samples
        batch_inds = np.random.choice(cifar_images.shape[0], b_s, replace=False)
        im_batch = cifar_images[batch_inds,...]
        # scale real images to (-1,1)
        im_batch = im_batch*2 -1
        # make some random noise
        noise_batch = np.random.uniform(-1,1,size=(b_s,2048))
        # Make uniform variable for gradient penalty calculation
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        # feed into discriminator training function and get
        # calculated error
        errD  = fn_trainD([noise_batch, im_batch, ϵ1])
    # record error for plotting
    dis_loss[ii] = errD
    
    # Train Generator using just the noise batch
    errG = fn_trainG([noise_batch])
    # record error for plotting
    gen_loss[ii] = errG
    # save every so often
    if ii % 100 == 0 and errG is not np.nan:
        GenModel.save_weights('CifarGANweights.h5')
    # Update progress bar to display error
    t.set_postfix(Dloss=dis_loss[ii],GLoss = gen_loss[ii])
# close progress bar when complete
t.close()

# The generator and discriminator are now finished training.
# We will plot the loss to see how it looks

# In Wasserstein training the discriminator loss
# is traditionally plotted negative so that it
# converges to 0 like a normal loss plot

# Display loss plots
plt.figure(figsize=(8,4))
plt.plot(np.arange(numIter),medfilt(-dis_loss[:,0],5),
         np.arange(numIter),medfilt(gen_loss[:,0],5))
plt.legend(['-Discriminator Loss',
            'Generator Loss'])
plt.show()


# Now display some test images
# First generate some noise
test_noise = np.random.uniform(-1,1,size=(16,2048))
# Get predictions on this noise and scale back to [0,1]
test_output = (GenModel.predict(test_noise) + 1)/2
test_output += 0
test_output[test_output<0] = 0
# Plot in subplots
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(test_output[i])
plt.show()