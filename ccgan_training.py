'''Libraries'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import sys
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from numpy.random import randn
from scipy.io import loadmat
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model,Sequential,Model
import math
import time
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Lambda,Concatenate,Dropout
import scipy
from keras.constraints import Constraint
from scipy.stats import truncnorm
from scipy.stats import weibull_min

#Parameters of the grid:
grid_size = 118 #300,1354
gen_size = 64
baseMVA = 100
solar_der_number = 40

'''Functions'''
'''
#This is the function to format the generators' cost
'''
def get_gencost(number_of_buses,mat_gen,mat_gencost):
  gen_ids = mat_gen[:,0]-1
  c1=np.zeros(number_of_buses)
  c2=np.zeros(number_of_buses)
  j=0
  for i in range(0,number_of_buses):
    if i in gen_ids:
      c1[i] = mat_gencost[j,5]
      c2[i] = mat_gencost[j,4]
      j=j+1
  tf_c1 = tf.convert_to_tensor(c1,dtype='float32')
  tf_c2 = tf.convert_to_tensor(c2,dtype='float32')
  return tf_c1,tf_c2

'''
#This is the function to get and format the upper and lower bound 
#of active&reactive power at each buse
'''
def get_pq_bound(number_of_buses, mat_gen):
  number_of_gens = mat_gen.shape[0]
  p_upper = np.zeros(number_of_buses)
  q_upper = np.zeros(number_of_buses)
  q_lower = np.zeros(number_of_buses)
  j=0
  for i in range(0,number_of_buses):
    if i in (mat_gen[:,0]-1):
      #pdb.set_trace()
      p_upper[i]=mat_gen[j,8]
      q_upper[i]=mat_gen[j,3]
      q_lower[i]=mat_gen[j,4]
      j=j+1
  tf_p_upper = tf.convert_to_tensor(p_upper/baseMVA,dtype='float32')
  tf_q_upper = tf.convert_to_tensor(q_upper/baseMVA,dtype='float32')
  tf_q_lower = tf.convert_to_tensor(q_lower/baseMVA,dtype='float32')
  return tf_p_upper,tf_q_upper,tf_q_lower

'''
#This is the function to get and format the upper and lower bound 
#of voltage magnitude&angle at each buse
'''
def get_vm_va_bound(mat_load):
  vm_lower = mat_load['aa'][:,-1]
  vm_upper = mat_load['aa'][:,-2]
  #vm_lower[0]=1
  #vm_upper[0]=1
  va_lower = np.ones(grid_size)*(-180)
  va_upper = np.ones(grid_size)*(180)
  #va_lower[0]=0
  #va_upper[0]=0
  tf_vm_upper = tf.convert_to_tensor(vm_upper,dtype='float32')
  tf_vm_lower = tf.convert_to_tensor(vm_lower,dtype='float32')
  tf_va_upper = tf.convert_to_tensor(va_upper,dtype='float32')
  tf_va_lower = tf.convert_to_tensor(va_lower,dtype='float32')
  return tf_vm_upper,tf_vm_lower,tf_va_upper,tf_va_lower

'''
#This is the function to define Model G
'''
def define_generator(latent_dim):
  code = keras.Input(shape=(1,1))
  random_noise = keras.Input(shape=(1,latent_dim))
  demand = keras.Input(shape=(1,grid_size*2))
  concat_layer= Concatenate(axis=-1)([code,random_noise, demand])
  hidden = Conv1D(256, (1), activation=None,padding='same')(concat_layer)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  #hidden = Dropout(0.1)(hidden)
  hidden = Conv1D(128, (1), activation=None,padding='same')(hidden)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  hidden = Conv1D(64, (1), activation=None,padding='same')(hidden)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  hidden = Conv1D(32, (1), activation=None,padding='same')(hidden)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  #hidden = Dropout(0.1)(hidden)
  hidden = layers.Flatten()(hidden)
  g_active = layers.Dense(grid_size,dtype='float32',activation='sigmoid')(hidden)
  g_active = tf.math.multiply(g_active,tf_p_upper)
  g_reactive = layers.Dense(grid_size,dtype='float32',activation='sigmoid')(hidden)
  g_reactive = tf.math.multiply(g_reactive,tf_q_upper-tf_q_lower)\
                                                    +tf_q_lower
  bus_vm = layers.Dense(grid_size,dtype='float32',activation='sigmoid')(hidden)
  bus_vm = tf.math.multiply(bus_vm,tf_vm_upper-tf_vm_lower)+tf_vm_lower
  bus_va = layers.Dense(grid_size,dtype='float32',activation='sigmoid')(hidden)
  bus_va = tf.math.multiply(bus_va,tf_va_upper-tf_va_lower)+tf_va_lower
  g_model = keras.Model(inputs=[code,random_noise,demand], outputs=[g_active,g_reactive,bus_vm,bus_va])
  return g_model

'''
#This is the function to define Model Q
'''
def define_qnetwork():
  g_active = keras.Input(shape=(1,grid_size))
  hidden = Conv1D(4, (1), activation=None,padding='same')(g_active)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  hidden = layers.Flatten()(hidden)
  out_layer = layers.Dense(1,dtype='float32',activation='tanh')(hidden)
  q_model = Model(g_active, out_layer)
  return q_model

'''
#This is the function to define Model D
'''
def define_gp_critic():
  demand = keras.Input(shape=(1,grid_size*2))
  g_active = keras.Input(shape=(1,grid_size))
  g_reactive = keras.Input(shape=(1,grid_size))
  bus_vm = keras.Input(shape=(1,grid_size))
  bus_va = keras.Input(shape=(1,grid_size))
  concat_layer= Concatenate(axis=-1)([demand,g_active,g_reactive,bus_vm,bus_va])
  hidden = Conv1D(512, (3), activation=None,padding='same')(concat_layer)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  hidden = layers.MaxPooling1D(pool_size=(3),padding='same')(hidden)
  hidden = Conv1D(256, (1), activation=None,padding='same')(hidden)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  hidden = Conv1D(128, (1), activation=None,padding='same')(hidden)
  hidden = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden)
  hidden1 = Conv1D(64, (1), activation=None,padding='same')(hidden)
  hidden1 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden1)
  hidden1 = Conv1D(32, (1), activation=None,padding='same')(hidden1)
  hidden1 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden1)
  hidden1 = Conv1D(16, (1), activation=None,padding='same')(hidden1)
  hidden1 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden1)
  hidden1 = Conv1D(8, (1), activation=None,padding='same')(hidden1)
  hidden1 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden1)
  hidden1 = layers.Flatten()(hidden1)
  hidden2 = Conv1D(64, (1), activation=None,padding='same')(hidden)
  hidden2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden2)
  hidden2 = Conv1D(32, (1), activation=None,padding='same')(hidden2)
  hidden2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden2)
  hidden2 = Conv1D(16, (1), activation=None,padding='same')(hidden2)
  hidden2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden2)
  hidden2 = Conv1D(8, (1), activation=None,padding='same')(hidden2)
  hidden2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden2)
  hidden2 = layers.Flatten()(hidden2)
  out_layer = layers.Dense(1,dtype='float32',activation=None)(hidden1)
  r_out_layer = layers.Dense(1,dtype='float32',activation=None)(hidden2)
  r1_out_layer = layers.Dense(1,dtype='float32',activation=None)(hidden2)
  #define model
  d_model = Model([demand,g_active,g_reactive,bus_vm,bus_va], [out_layer,r_out_layer,r1_out_layer])
  return d_model


'''
#This is the function to define GAN
'''
def define_gan(g_model, d_model):
  # make weights in the discriminator not trainable
  d_model.trainable = False
  [g_active,g_reactive,bus_vm,bus_va] = generator.output
  [gen_code, gen_noise, gen_label] = generator.input
  gan_output = d_model([tf.reshape(gen_label,[-1,1,grid_size*2]),tf.reshape(g_active,[-1,1,grid_size]),\
                        tf.reshape(g_reactive,[-1,1,grid_size]),tf.reshape(bus_vm,[-1,1,grid_size]),\
                        tf.reshape(bus_va,[-1,1,grid_size])])
  # define gan model as taking noise and label and outputting a classification
  model = Model([gen_code, gen_noise, gen_label], gan_output)
  return model

'''
#This is the function to define GAN
'''
def define_ae(g_model,q_model):
  [g_active,g_reactive,bus_vm,bus_va] = g_model.output
  [gen_code, gen_noise, gen_label] = g_model.input
  #realp_tf = g_active*baseMVA
  #cost1=tf.math.multiply(realp_tf, tf_c1)
  #cost2=tf.math.multiply(K.square(realp_tf), tf_c2)
  q_output = q_model(tf.reshape(g_active,[-1,1,grid_size]))
  ae_model = Model([gen_code, gen_noise, gen_label], q_output)
  return ae_model

'''
#This is the function to sample P/Q demand in positive/negative 20 percent
'''
def rdm_load_pq_20per(case_pload,case_qload,sample_number):
    number_of_bus = case_pload.shape[0]
    p_demand = np.zeros([sample_number,number_of_bus])
    q_demand = np.zeros([sample_number,number_of_bus])
    for i in range(0,number_of_bus):
        pi = case_pload[i]
        qi = case_qload[i]
        if case_pload[i]!=0:
          p_mw = np.random.uniform(pi*0.8,pi*1.2,sample_number)
          q_mvar = np.random.uniform(qi*0.8,qi*1.2,sample_number)
          p_demand[:,i] = p_mw
          q_demand[:,i] = q_mvar
    return p_demand,q_demand

'''
#This is the function to generate three latent variables
'''
def generate_latent_points(latent_dim, n_samples):
  # generate points in the latent space
  code_input = np.random.uniform(-1,1,n_samples).reshape([n_samples,1])
  x_input = randn(latent_dim * n_samples)
  z_input = x_input.reshape(n_samples, latent_dim)
  # generate labels
  p_demand,q_demand = rdm_load_pq_20per(case118_pload,case118_qload,n_samples)
  labels = np.concatenate([p_demand,q_demand],axis=1)
  return [code_input,z_input, labels]

'''
#This is the function to draw real scenarios
'''
def generate_real_samples(n_samples):
  # choose random instances
  ix = np.random.randint(0, condition.shape[0], n_samples)
  # select images and labels
  batch_condition,batch_vm,batch_va,batch_p,batch_q = condition[ix], \
    real_vm[ix], real_va[ix], real_p[ix], real_q[ix]
  # generate class labels
  y = -np.ones((n_samples, 1))
  return batch_condition,batch_vm,batch_va,batch_p,batch_q, y

'''
# use the generator to generate n fake examples, with class labels
'''
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	c_input, z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([c_input, z_input, labels_input])
	# create class labels
	y = np.ones((n_samples, 1))
	return c_input, labels_input,images, y

'''
# Checking feasibility based on PF equations
'''
def feasibility_checking(p_demand,q_demand, active_p,reactive_q,vm,va):
  generated_size = p_demand.shape[0]
  #Step1: calculate power withdraw on each bus
  P_out = tf.convert_to_tensor(p_demand-active_p,dtype='float32')
  Q_out = tf.convert_to_tensor(q_demand-reactive_q,dtype='float32')
  #pdb.set_trace()
  #get voltage on each bus
  v_r = tf.math.multiply(vm,tf.cos(tf.math.multiply(va,tf.constant(math.pi/180,dtype='float32'))))
  v_i = tf.math.multiply(vm,tf.sin(tf.math.multiply(va,tf.constant(math.pi/180,dtype='float32'))))
  V = tf.reshape(tf.complex(v_r,v_i),[generated_size,118])
  #calculate current
  Y_bus_tf = tf.convert_to_tensor(Y_bus)
  I = tf.matmul(V,Y_bus_tf)
  #calculate power injection on each bus
  S_in = tf.math.multiply(V,tf.math.conj(I))
  P_in = tf.math.real(S_in)
  Q_in = tf.math.imag(S_in)
  #evaluate the balance
  p_balance = tf.reduce_mean(P_in+P_out,axis=0)
  q_balance = tf.reduce_mean(Q_in+Q_out,axis=0)
  return p_balance,q_balance

'''
# use the generator to generate n fake examples, with class labels
'''
def generate_test_samples(generator, latent_dim, test_samples):
  # generate points in latent space
  n_samples = opf_real_conditions_validation.shape[0]
  z_input, labels_input = generate_latent_points(latent_dim, n_samples)
  # predict outputs
  images = generator.predict([z_input, opf_real_conditions_validation])
  # create class labels
  y = np.ones((n_samples, 1))
  return opf_real_conditions_validation,images, y

'''
# Checking optimality of validation data
'''
def optimality_checking(p_opt,q_opt,p_fake,q_fake):
  p_mse = K.mean(K.square(p_opt-p_fake))
  q_mse = K.mean(K.square(q_opt-q_fake))
  return p_mse,q_mse

  
'''
#This is the block to load data from matlab, and csv dataset
#Then process them ready for the neural network training
'''
#Load grid data from matlab, these 5 mat data are Ybus, MPC.gen, MPC.bus, MPC.gencost and MPC.branches
mat_y = loadmat('Y_bus.mat')
mat_gen = loadmat('grid_gen.mat')['gen']
mat_load = loadmat('grid_load.mat')
mat_gencost = loadmat('grid_gencost.mat')['gencost']
line_constraints = loadmat('line_constraints.mat')['line_constraint']

#Load dataset, generated by Matpower PF
condition = loadmat('condition.mat')['conditions_list']
solution = loadmat('solution.mat')['datapoints_list']

#Validation set, generated by Matpower OPF
opf_condition = loadmat('validation_condition.mat')['conditions_list']
opf_points = loadmat('validation_solution.mat')['datapoints_list']

#Data preprocessing
Y_bus = mat_y['Y_bus'].toarray().astype('complex64')
default_pload = mat_load['load'][:,2]/baseMVA
default_qload = mat_load['load'][:,3]/baseMVA
real_vm = solution[:,0:grid_size]
real_va = solution[:,grid_size:grid_size*2]
matpower_p = real_points[:,grid_size*2:grid_size*2+gen_size]
matpower_q = real_points[:,grid_size*2+gen_size:]
real_p = np.zeros(real_vm.shape)
real_q = np.zeros(real_vm.shape)
gen_ids = mat_gen[:,0]-1
j=0
for i in range(0,grid_size):
  if i in gen_ids:
    real_p[:,i]=matpower_p[:,j]
    real_q[:,i]=matpower_q[:,j]
    j=j+1
real_p=real_p/baseMVA
real_q = real_q/baseMVA
tf_p_upper,tf_q_upper,tf_q_lower = get_pq_bound(grid_size, mat_gen)
tf_vm_upper,tf_vm_lower,tf_va_upper,tf_va_lower = get_vm_va_bound(mat_load)
tf_c1,tf_c2 = get_gencost(grid_size,mat_gen,mat_gencost)
gen_id=mat_gen[:,0]-1
non_gen_ids = np.zeros([gen_size])
j=0
for i in range(0,grid_size):
  if i not in gen_id:
    non_gen_ids[j]=i
    j=j+1
wind_der_ids = non_gen_ids[:(-1)*solar_der_number]
solar_der_ids = non_gen_ids[(-1)*1olar_der_number:]

line_constraints.shape
from_buses = line_constraints[:,1]-1
to_buses = line_constraints[:,2]-1
s_max = line_constraints[:,4]

constraint_matrix = np.zeros([grid_size,grid_size])
i=0
for fb in from_buses:
  constraint_matrix[int(fb),int(to_buses[i])]=s_max[i]
  i=i+1


#Define models
latent_dim = 200
batch_size = 512
half_batch = batch_size//2
# Define all networks
discriminator = define_gp_critic()
generator =define_generator(latent_dim)
qnetwork = define_qnetwork()
gan_model = define_gan(generator, discriminator)
ae_model = define_ae(generator,qnetwork)


#Model training
iterations = 100000
#time scale of model D
n_critic=10
#weight for each loss component
lbd=10
mu=10
delta=100
nu=1000
xi=10
#Training begins
for i in range(iterations):
  with tf.GradientTape(persistent=True) as tape:
    discriminator.trainable=True
    #Initializing optimizers:
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    r_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)  
    '''
      sub-block 1 starts: Training the discriminator for $n_critic times
    '''
    for _j in range(n_critic):
      #Prepare real and fake data
      real_batch_demand, real_batch_vm,real_batch_va,real_batch_p,real_batch_q, \
                  real_batch_y = generate_real_samples(half_batch)
      [fake_latent_code, fake_latent_noise, _] = \
                  generate_latent_points(latent_dim, half_batch)
      [fake_batch_p,fake_batch_q,fake_batch_vm,fake_batch_va] = \
                  generator([tf.reshape(fake_latent_code,[-1,1,1]),\
                  tf.reshape(fake_latent_noise,[-1,1,latent_dim]),\
                  tf.reshape(real_batch_demand,[-1,1,118*2])])
      fake_batch_y = np.ones((half_batch, 1))
      alpha = K.random_uniform(
          shape=[half_batch,1], 
          minval=0.,
          maxval=1.
      )
      diff_batch_vm = real_batch_vm - fake_batch_vm
      diff_batch_va = real_batch_va - fake_batch_va
      diff_batch_p = real_batch_p - fake_batch_p
      diff_batch_q = real_batch_q - fake_batch_q

      inter_batch_vm = real_batch_vm + (alpha*diff_batch_vm)
      inter_batch_va = real_batch_va + (alpha*diff_batch_va)
      inter_batch_p = real_batch_p + (alpha*diff_batch_p)
      inter_batch_q = real_batch_q + (alpha*diff_batch_q)

      '''
      Calculate the Cost function for discriminator
      '''
      #Prepare the gradient penalty
      inter_d_output,_,_ = discriminator([tf.reshape(real_batch_demand,[-1,1,118*2]),\
                                tf.reshape(inter_batch_p,[-1,1,118]),\
                                tf.reshape(inter_batch_q,[-1,1,118]),\
                                tf.reshape(inter_batch_vm,[-1,1,118]),\
                                tf.reshape(inter_batch_va,[-1,1,118])])
      inter_gradients_p = tape.gradient(inter_d_output, [inter_batch_p,inter_batch_q,inter_batch_vm,inter_batch_va])
      rdc_sum = tf.math.reduce_sum(tf.math.reduce_sum(tf.square(inter_gradients_p),axis=0),axis=1)
      slopes = K.sqrt(rdc_sum)
      gradient_penalty = tf.reduce_mean(tf.square((slopes-1.)))
      
      #calculate the regular WGAN loss
      real_d_output,_,_ = discriminator([tf.reshape(real_batch_demand,[-1,1,118*2]),\
                                            tf.reshape(real_batch_p,[-1,1,118]),\
                                            tf.reshape(real_batch_q,[-1,1,118]),\
                                            tf.reshape(real_batch_vm,[-1,1,118]),\
                                            tf.reshape(real_batch_va,[-1,1,118])])
      fake_d_output,_,_ = discriminator([tf.reshape(real_batch_demand,[-1,1,118*2]),\
                                            tf.reshape(fake_batch_p,[-1,1,118]),\
                                            tf.reshape(fake_batch_q,[-1,1,118]),\
                                            tf.reshape(fake_batch_vm,[-1,1,118]),\
                                            tf.reshape(fake_batch_va,[-1,1,118])])
      d1_loss = K.mean(real_d_output*real_batch_y)
      d2_loss = K.mean(fake_d_output*fake_batch_y)
      d_loss = d1_loss+d2_loss+delta*gradient_penalty
      
      #-----------Feasibility Score Part-------------
      #Step0: Select some real data
      if _j%n_critic==0:
        real_data_num_for_r = 3
        downnsample_batch_demand, downnsample_real_batch_vm,downnsample_real_batch_va,\
        downnsample_real_batch_p,downnsample_real_batch_q,downnsample_real_batch_y \
        = generate_real_samples(real_data_num_for_r)
        r_batch_demand = tf.concat([real_batch_demand,downnsample_batch_demand],axis=0)
        r_batch_vm = tf.concat([fake_batch_vm,downnsample_real_batch_vm],axis=0)
        r_batch_va = tf.concat([fake_batch_va,downnsample_real_batch_va],axis=0)
        r_batch_p = tf.concat([fake_batch_p,downnsample_real_batch_p],axis=0)
        r_batch_q = tf.concat([fake_batch_q,downnsample_real_batch_q],axis=0)
        #Step1: calculate power withdraw on each bus
        PQ_out = tf.convert_to_tensor(tf.dtypes.cast(r_batch_demand, tf.float32)-tf.concat([r_batch_p,\
                                                                      r_batch_q],axis=1),dtype='float32')
        #get voltage on each bus
        v_r = tf.math.multiply(r_batch_vm,tf.cos(tf.math.multiply(r_batch_va,\
                                                                    tf.constant(math.pi/180,dtype='float32'))))
        v_i = tf.math.multiply(r_batch_vm,tf.sin(tf.math.multiply(r_batch_va,\
                                                                    tf.constant(math.pi/180,dtype='float32'))))
        V = tf.reshape(tf.complex(v_r,v_i),[-1,number_of_buses])
        #calculate current
        Y_bus_tf = tf.convert_to_tensor(Y_bus)
        I = tf.matmul(V,Y_bus_tf)
        #calculate power injection on each bus
        S_in = tf.math.multiply(V,tf.math.conj(I))
        P_in = tf.math.real(S_in)
        Q_in = tf.math.imag(S_in)
        PQ_balance = tf.concat([P_in,Q_in],axis=1)+PQ_out
        mean_PQ_balance = tf.reshape(tf.reduce_mean(K.abs(PQ_balance),axis=1),[-1,1])
        _,r_output,r2_output = discriminator([tf.reshape(r_batch_demand,[-1,1,grid_size*2]),\
                                            tf.reshape(r_batch_p,[-1,1,grid_size]),\
                                            tf.reshape(r_batch_q,[-1,1,grid_size]),\
                                            tf.reshape(r_batch_vm,[-1,1,grid_size]),\
                                            tf.reshape(r_batch_va,[-1,1,grid_size])])
        r_loss = tf.reduce_mean(K.square(mean_PQ_balance-r_output))
        v_r = tf.math.multiply(fake_batch_vm,tf.cos(tf.math.multiply(fake_batch_va,\
                                                                 tf.constant(math.pi/180,dtype='float32'))))
        v_i = tf.math.multiply(fake_batch_vm,tf.sin(tf.math.multiply(fake_batch_va,\
                                                                    tf.constant(math.pi/180,dtype='float32'))))
        fake_v = tf.reshape(tf.complex(v_r,v_i),[-1,grid_size])
        V_ij_conjugate=tf.math.conj(tf.reshape(fake_v,[-1,grid_size,1])-tf.reshape(fake_v,[-1,1,grid_size]))
        V_Vij_conjugate = tf.multiply(tf.reshape(fake_v,[-1,grid_size,1]),V_ij_conjugate)
        broad_Ybus = tf.reshape(Y_bus_tf,[1,118,118])
        flow = tf.multiply(V_Vij_conjugate,tf.math.conj(broad_Ybus*(-1)))
        flow_norm = K.sqrt(K.square(tf.math.real(flow)*100)+K.square(tf.math.imag(flow)*100))
        branch_flow_loss = tf.reduce_mean(tf.nn.relu(flow_norm-constraint_matrix))
        r2_loss = tf.reduce_mean(K.square(branch_flow_loss-r2_output))
        d_grads = tape.gradient(d_loss+xi*(r_loss+r2_loss),discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(d_grads+d_grads, discriminator.trainable_weights))
    '''
      Sub-block 1 ends.
    '''
    '''
      Block 2 starts: Training the generator
    '''
    #Train Generator 
    full_real_batch_demand,_,_,_,_,_ = generate_real_samples(batch_size)
    [fake_latent_code, fake_latent_noise, _] = \
                  generate_latent_points(latent_dim, batch_size)
    tf_fake_batch_demand = tf.keras.backend.variable(full_real_batch_demand)
    tape.watch(tf_fake_batch_demand)
    [fake_batch_p,fake_batch_q,fake_batch_vm,fake_batch_va] = \
                        generator([tf.reshape(fake_latent_code,[-1,1,1]),\
                        tf.reshape(fake_latent_noise,[-1,1,latent_dim]),\
                        tf.reshape(tf_fake_batch_demand,[-1,1,grid_size*2])])
    fake_gan_output,_,_ = discriminator([tf.reshape(tf_fake_batch_demand,[-1,1,grid_size*2]),\
                                            tf.reshape(fake_batch_p,[-1,1,grid_size]),\
                                            tf.reshape(fake_batch_q,[-1,1,grid_size]),\
                                            tf.reshape(fake_batch_vm,[-1,1,grid_size]),\
                                            tf.reshape(fake_batch_va,[-1,1,grid_size])])
    fake_batch_y = -np.ones((batch_size, 1))
    gan_loss = K.mean(fake_gan_output*fake_batch_y)
    dy_dx = tf.reduce_mean(tf.square(tape.gradient([fake_batch_p,fake_batch_q,\
                                                    fake_batch_vm,fake_batch_va],\
                                                    tf_fake_batch_demand)))
    #-----------physics
    #Step1: calculate power withdraw on each bus
    PQ_out = tf.convert_to_tensor(tf_fake_batch_demand-tf.concat([fake_batch_p,\
                                                                  fake_batch_q],axis=1),dtype='float32')
    #get voltage on each bus
    v_r = tf.math.multiply(fake_batch_vm,tf.cos(tf.math.multiply(fake_batch_va,\
                                                                 tf.constant(math.pi/180,dtype='float32'))))
    v_i = tf.math.multiply(fake_batch_vm,tf.sin(tf.math.multiply(fake_batch_va,\
                                                                 tf.constant(math.pi/180,dtype='float32'))))
    V = tf.reshape(tf.complex(v_r,v_i),[-1,number_of_buses])
    #calculate current
    Y_bus_tf = tf.convert_to_tensor(Y_bus)
    I = tf.matmul(V,Y_bus_tf)
    #calculate power injection on each bus
    S_in = tf.math.multiply(V,tf.math.conj(I))
    P_in = tf.math.real(S_in)
    Q_in = tf.math.imag(S_in)
    PQ_balance = tf.concat([P_in,Q_in],axis=1)+PQ_out
    mean_PQ_balance = tf.reduce_mean(K.square(PQ_balance))
    V_ij_conjugate=tf.math.conj(tf.reshape(V,[-1,grid_size,1])-tf.reshape(V,[-1,1,grid_size]))
    V_Vij_conjugate = tf.multiply(tf.reshape(V,[-1,grid_size,1]),V_ij_conjugate)
    broad_Ybus = tf.reshape(Y_bus_tf,[1,grid_size,grid_size])
    flow = tf.multiply(V_Vij_conjugate,tf.math.conj(broad_Ybus*(-1)))
    flow_norm = K.sqrt(K.square(tf.math.real(flow)*100)+K.square(tf.math.imag(flow)*100))
    branch_flow_loss = tf.reduce_mean(tf.nn.relu(flow_norm-constraint_matrix))
    #p_balance = tf.reduce_mean(P_in+P_out,axis=0)
    #q_balance = tf.reduce_mean(Q_in+Q_out,axis=0)
    discriminator.trainable=False
    gan_grads = tape.gradient(gan_loss+lbd*dy_dx+mu*(mean_PQ_balance+branch_flow_loss),gan_model.trainable_weights)
    g_optimizer.apply_gradients(zip(gan_grads, gan_model.trainable_weights))
  
    #Training the Q Inference Network
    q_output = qnetwork(tf.reshape(fake_batch_p,[-1,1,grid_size]))
    q_loss = nu*K.mean(K.square(fake_latent_code-q_output))
    ae_loss_list.append(q_loss.numpy())
    ae_grads = tape.gradient(q_loss, ae_model.trainable_weights)
    q_optimizer.apply_gradients(zip(ae_grads, ae_model.trainable_weights))
    '''
      Sub-block 2 ends.
    '''
