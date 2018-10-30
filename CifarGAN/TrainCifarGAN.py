
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, Dense, Reshape
from keras.layers import LeakyReLU, BatchNormalization, UpSampling2D
from keras.layers import concatenate, GlobalAveragePooling2D
from keras.models import Model
from keras.initializers import RandomNormal, he_normal, glorot_uniform
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam
import keras.backend as K
from tqdm import tqdm, trange
from scipy.signal import medfilt
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt


# In[2]:


(cifar_images, _), (_, _) = cifar10.load_data()
cifar_images = (cifar_images)/255


# In[3]:


plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(cifar_images[np.random.randint(0,50000)])
plt.show()


# Weight initializers

# In[4]:


# Random Normal initializer
# init1 = RandomNormal(0, 0.01)
init1 = glorot_uniform()
init2 = glorot_uniform()


# Create GAN generator model

# In[5]:


# input layer that accepts our 1D input noise vectors
g_input = Input(shape=(2048,),name='NoiseInput')
# reshape for use in convolutional layers
x = Reshape((4,4,128))(g_input)
# first deconvolutional layer
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

# In[6]:


GenModel.summary()


# Create GAN discriminator model using an InceptionV3 format

# In[7]:


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

# In[8]:


DisModel.summary()


# Create training functions

# In[9]:


lrD = 1e-7  # discriminator learning rate
lrG = 1e-6  # generator learning rate

#%% Setup training graph
noise_input = GenModel.inputs[0]
fake_output = GenModel.outputs[0]
real_output = DisModel.inputs[0]

# noise-to-image generator function
fn_genIm = K.function([noise_input],[fake_output])
# discriminator scores
realImScore = DisModel([real_output])
fakeImScore = DisModel([fake_output])
# create mixed output for gradient penalty
ep_input = K.placeholder(shape=(None,1,1,1))
mixed_output = Input(shape=(32,32,3),
                    tensor=ep_input * real_output + (1-ep_input) * fake_output)
mixed_score = DisModel([mixed_output])
# discriminator losses
realDloss = K.mean(realImScore)
fakeDloss = K.mean(fakeImScore)
# gradient penalty loss
grad_mixed = K.gradients([mixed_score],[mixed_output])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_D = fakeDloss - realDloss + 1 * grad_penalty

#%% Image generator loss
# variational loss + discriminator penalty
var_loss= -K.mean(K.std(fake_output,axis=0))
loss_G = .01*var_loss-fakeDloss

#%% Training functions
# Discriminator training function
# D_trups = SGD(lr=lrD,momentum=0.9,nesterov=True).get_updates(DisModel.trainable_weights,[],loss_D)
D_trups = Adam(lr=lrD).get_updates(DisModel.trainable_weights,[],loss_D)
fn_trainD = K.function([noise_input, real_output, ep_input],[loss_D], D_trups)

# Generator Training function
# G_trups = SGD(lr=lrG,momentum=0.9,nesterov=True).get_updates(GenModel.trainable_weights,[],loss_G)
G_trups = Adam(lr=lrG).get_updates(GenModel.trainable_weights,[],loss_G)
fn_trainG = K.function([noise_input], [loss_G], G_trups)


# In[10]:


fn_evalD = K.function([noise_input,real_output,ep_input],[loss_D,fakeDloss,realDloss,grad_penalty])


# In[11]:


# set number of iterations to do
numIter = 40000
# set batch size
b_s = 64
# preallocate for the training and validation losses
dis_loss = np.zeros((numIter,1))
gen_loss = np.zeros((numIter,1))


# In[12]:


# pre-train disciminator a little
for _ in range(50):
    # Train Discriminator
    # grab random training samples
    batch_inds = np.random.choice(cifar_images.shape[0], b_s, replace=False)
    im_batch = cifar_images[batch_inds,...]
    # scale real images to (-1,1)
    im_batch = im_batch*2 -1
    # make some random noise
    noise_batch = np.random.uniform(-1,1,size=(b_s,2048))
    # train discrimators
    ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
    errD  = fn_trainD([noise_batch, im_batch, ϵ1])


# In[13]:


# Delete progress bar variable and reset
if 't' in locals():
    t.close()
    del t
t = trange(numIter,file=sys.stdout)
# Main training loop
for ii in t:
    for _ in range(5):
        # Train Discriminator
        # grab random training samples
        batch_inds = np.random.choice(cifar_images.shape[0], b_s, replace=False)
        im_batch = cifar_images[batch_inds,...]
        # scale real images to (-1,1)
        im_batch = im_batch*2 -1
        # make some random noise
        noise_batch = np.random.uniform(-1,1,size=(b_s,2048))
        # train discrimators
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = fn_trainD([noise_batch, im_batch, ϵ1])
    dis_loss[ii] = errD
    
    # Train Generator
    errG = fn_trainG([noise_batch])
    gen_loss[ii] = errG
    # save every so often
    if ii % 100 == 0 and errG is not np.nan:
        GenModel.save_weights('CifarGANweights.h5')
    # Update progress bar
    t.set_postfix(Dloss=dis_loss[ii],GLoss = gen_loss[ii])
    
t.close()


# In[14]:


# grab some real images for testing disciminator
batch_inds = np.random.choice(cifar_images.shape[0], b_s, replace=False)
im_batch = cifar_images[batch_inds,...]
# scale real images to (-1,1)
im_batch = im_batch*2 -1
# make some random noise
noise_batch = np.random.uniform(-1,1,size=(b_s,2048))
# train discrimators
ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
Dresults  = fn_evalD([noise_batch, im_batch, ϵ1])
print('D loss is:',Dresults[0])
print('Fake score is: ',Dresults[1])
print('Real score is: ',Dresults[2])
print('Gradient penalty is:',Dresults[3])


# In[ ]:


# Display loss plots
plt.figure(figsize=(8,4))
plt.plot(np.arange(numIter),medfilt(-dis_loss[:,0],5),
         np.arange(numIter),medfilt(gen_loss[:,0],5))
plt.legend(['-Discriminator Loss',
            'Generator Loss'])
# plt.ylim([-1,.5]);


# In[ ]:


# display some test images
test_noise = np.random.uniform(-1,1,size=(16,2048))
# test_output = (fn_genIm([test_noise])[0] + 1)/2
test_output = (GenModel.predict(test_noise) + 1)/2
test_output += 0
test_output[test_output<0] = 0
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(test_output[i])
plt.show()

