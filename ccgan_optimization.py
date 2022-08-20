#Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import sys
import pandas as pd
from pandas import DataFrame as df
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from numpy.random import randn
from scipy.io import loadmat
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model,Sequential,Model
import math
import time
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Lambda,Concatenate
import scipy
from keras.constraints import Constraint
import datetime
from scipy.stats import weibull_min

#Parameters
grid_size=118
gen_size=64
baseMVA = 100

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
#Checking optimality
'''
def optimality_checking(p_opt,q_opt,p_fake,q_fake):
  #pdb.set_trace()
  p_mse = K.sum(K.square(p_opt*100-p_fake*100))
  q_mse = K.sum(K.square(q_opt*100-q_fake*100))
  return p_mse,q_mse

'''
#Checking cost optimality
'''
def financial_optimality_checking(p_opt,q_opt,p_fake,q_fake):
  #pdb.set_trace()
  p_opt_actual = tf.convert_to_tensor(p_opt*100,dtype=tf.float32)
  p_fake_actual = tf.convert_to_tensor(p_fake*100,dtype=tf.float32)
  #pdb.set_trace()
  p_opt_actual_cost = K.sum(tf.math.multiply(p_opt_actual, tf_c1)+\
                              tf.math.multiply(K.square(p_opt_actual), tf_c2))
  p_fake_actual_cost = K.sum(tf.math.multiply(p_fake_actual, tf_c1)+\
                            tf.math.multiply(K.square(p_fake_actual), tf_c2))
  p_mse = K.sum(p_opt_actual_cost-p_fake_actual_cost)
  print(p_opt_actual_cost)
  print(p_fake_actual_cost)
  print(p_mse)
  return p_mse

'''
#Checking feasibility
'''
def feasibility_checking(p_demand,q_demand, active_p,reactive_q,vm,va):
  #pdb.set_trace()
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

def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

def replaceinf(t):
    return tf.where(tf.math.is_inf(t), tf.zeros_like(t), t)

  

#Load grid data from matlab, these 5 mat data are Ybus, MPC.gen, MPC.bus, MPC.gencost and MPC.branches
mat_y = loadmat('Y_bus.mat')
mat_gen = loadmat('grid_gen.mat')['gen']
mat_load = loadmat('grid_load.mat')
mat_gencost = loadmat('grid_gencost.mat')['gencost']
line_constraints = loadmat('line_constraints.mat')['line_constraint']

#Test set, generated by Matpower OPF
opf_condition = loadmat('test_condition.mat')['conditions_list']
opf_points = loadmat('test_solution.mat')['datapoints_list']

gen_ids=mat_gen[:,0]-1
default_pload = mat_load['load'][:,2]/baseMVA
default_qload = mat_load['load'][:,3]/baseMVA
opf_real_conditions = opf_condition[:]/baseMVA
opf_real_vm= opf_points[:,0:grid_size]
opf_real_va = opf_points[:,grid_size:grid_size*2]
opf_matpower_pq = opf_points[:,grid_size*2:]
opf_matpower_p = opf_points[:,grid_size*2:grid_size*2+gen_size]
opf_matpower_q = opf_points[:,grid_size*2+gen_size:]
opf_real_p = np.zeros([opf_points.shape[0],grid_size])
opf_real_q = np.zeros([opf_points.shape[0],grid_size])
j=0
for i in range(0,grid_size):
  if i in gen_ids:
    opf_real_p[:,i]=opf_matpower_p[:,j]
    opf_real_q[:,i]=opf_matpower_q[:,j]
    j=j+1
opf_real_p=opf_real_p/baseMVA
opf_real_q=opf_real_q/baseMVA

gen_id=mat_gen[:,0]-1
basis_vector = np.zeros(grid_size)
for i in range(0,grid_size):
  if i in gen_ids:
    basis_vector[i]=1

'''Load saved models'''
latent_dim = 200 #300
discriminator = load_model('d.h5')
generator = load_model('g.h5')

'''Plot the representational curve of latent code c'''
number=0
x = np.arange(-1,1,0.01)
z = np.zeros([200])
c = np.zeros([200])
tf_c1,tf_c2=get_gencost(grid_size,mat_gen,mat_gencost)
c_input, z_input, labels_input = generate_latent_points(latent_dim, 1)
for i in range(0,200):
    c_input[0,0]=x[i]
    images = generator.predict([c_input.reshape([-1,1,1]),z_input.reshape([-1,1,latent_dim]),\
                                opf_real_conditions[number,:].reshape([1,1,grid_size*2])])
    op=financial_optimality_checking(opf_real_p[number,:],opf_real_q[number,:],images[0],images[1])
    z[i]=op
    cost1=tf.math.multiply(images[0]*100, tf_c1)
    cost2=tf.math.multiply(K.square(images[0]*100), tf_c2)
    print(1)
    print(tf.reduce_sum(cost2))
    suma = tf.reduce_sum(cost1+cost2,axis=1)
    c[i]=suma
    print(str(i))
plt.plot(c[:])
plt.xlabel('Value of the Latent Code')
plt.ylabel('Cost Value')
plt.xticks(np.arange(0, 200, 20), np.arange(-10, 10, 2)/10)


'''Optimization begins'''
tf_p_upper,tf_q_upper,tf_q_lower = get_pq_bound(grid_size, mat_gen)
#Sample numeber
sample_batch = 5000 #100,200,500,1000,5000
#Cost coefficients
tf_c1,tf_c2=get_gencost(grid_size,mat_gen,mat_gencost)
#for n test cases
for test_iter in range(0,number_of_points):
  p_opt_actual = tf.convert_to_tensor(opf_real_p[test_iter,:]*100,dtype=tf.float32)
  cost1=tf.math.multiply(p_opt_actual, tf_c1)
  cost2=tf.math.multiply(K.square(p_opt_actual), tf_c2)
  suma = tf.reduce_sum(cost1+cost2)
  k=0
  number=test_iter
  #Generate latent points
  c_input, z_input, _ = generate_latent_points(latent_dim, sample_batch)
  tf_z=tf.convert_to_tensor(z_input.reshape([sample_batch,1,latent_dim]))
  images = generator.predict([c_input.reshape([-1,1,1]),tf_z,np.repeat(opf_real_conditions[number,:].reshape([-1,1,grid_size*2]),sample_batch,axis=0)])
  y_a= K.sum(tf.math.multiply(images[0]*100, tf_c1)+tf.math.multiply(K.square(images[0]*100), tf_c2),axis=1)
  solutions = images[0]
  #Select best candidates
  _,score,score2 = discriminator([np.repeat(opf_real_conditions[number,0:grid_size*2].reshape([-1,1,grid_size*2]),sample_batch,axis=0),\
                                          tf.reshape(images[0],[-1,1,grid_size]),\
                                          tf.reshape(images[1],[-1,1,grid_size]),\
                                          tf.reshape(images[2],[-1,1,grid_size]),\
                                          tf.reshape(images[3],[-1,1,grid_size])])
  candidates =y_a.numpy()
  best_index = np.argmin(score+score2)
  best_score1 = score[best_index]
  best_score2 = score2[best_index]
  if best_score1+best_score2>zeta:
    continue
  P = images[0]
  tf_p = images[0]
  tf_q = images[1]
  tf_v = images[2]
  tf_phi = images[3]
  best_p = tf.reshape(tf_p[best_index],[-1,1,grid_size])
  best_q = tf.reshape(tf_q[best_index],[-1,1,grid_size])
  best_v = tf.reshape(tf_v[best_index],[-1,1,grid_size])
  best_phi = tf.reshape(tf_phi[best_index],[-1,1,grid_size])
  dual_v = np.zeros(1)
  dual_beta = np.zeros([1,1,grid_size])
  dual_gamma = np.zeros([1,1,grid_size])
  dual_epsilon = np.zeros(1)
  g_sign = np.zeros([1,1,grid_size])
  p_gradient = np.zeros([1,1,grid_size])
  v_gradient = np.zeros([1,1,grid_size])
  update_dim = np.ones([1,1,grid_size])
  #Update power dispatches
  for l in range(0,k_max):
    h=0.00000001
    best_p = best_p-0.0000001*p_gradient*update_dim
    best_p_plus = np.repeat(tf.reshape(best_p,[-1,1,grid_size]),grid_size,axis=0)+(np.identity(grid_size)*basis_vector).reshape([-1,1,grid_size])*h
    best_p_minus = np.repeat(tf.reshape(best_p,[-1,1,grid_size]),grid_size,axis=0)-(np.identity(grid_size)*basis_vector).reshape([-1,1,grid_size])*h
    _,score_plus,score_plus2 = discriminator([np.repeat(opf_real_conditions[number,0:grid_size*2].reshape([-1,1,grid_size*2]),grid_size,axis=0),\
                            best_p_plus,np.repeat(tf.reshape(best_q,[-1,1,grid_size]),grid_size,axis=0),\
                            np.repeat(tf.reshape(best_v,[-1,1,grid_size]),grid_size,axis=0),\
                            np.repeat(tf.reshape(best_phi,[-1,1,grid_size]),grid_size,axis=0)])
    _,score_minus,score_minus2 = discriminator([np.repeat(opf_real_conditions[number,0:grid_size*2].reshape([-1,1,grid_size*2]),grid_size,axis=0),\
                            best_p_minus,np.repeat(tf.reshape(best_q,[-1,1,grid_size]),grid_size,axis=0),\
                            np.repeat(tf.reshape(best_v,[-1,1,grid_size]),grid_size,axis=0),\
                            np.repeat(tf.reshape(best_phi,[-1,1,grid_size]),grid_size,axis=0)])
    dj = tf.reshape((score_plus-score_minus),[1, 1, grid_size])/(h*2)
    dj2 = tf.reshape((score_plus2-score_minus2),[1, 1, grid_size])/(h*2)
    p_gradient = 20000*tf.math.multiply(best_p, tf_c2)+tf_c1*100+dual_epsilon*dj2+dual_v*dj+dual_beta-dual_gamma
    if l==0:
      g_sign = np.sign(p_gradient)
      np.place(g_sign, g_sign==0., 2)
    if np.all(np.sign(p_gradient)!=g_sign):
      break
    else:
      update_dim = np.sign(p_gradient)==g_sign
    best_p = tf.math.minimum(tf.math.maximum(best_p,tf.zeros([1,1,grid_size])),tf.reshape(tf_p_upper,[1,1,grid_size]))
    lagrangian = tf.reduce_sum(tf.math.multiply(best_p*100, tf_c1)+tf.math.multiply(K.square(best_p*100), tf_c2)+\
        dual_v*best_score1+dual_beta*(best_p-tf_p_upper)-dual_gamma*best_p+dual_epsilon*best_score2)
    dual_v = dual_v+lagrangian/best_score1
    dual_epsilon = dual_v+lagrangian/best_score2
    dual_beta = tf.nn.relu(dual_beta+replaceinf(replacenan(lagrangian/(best_p-tf_p_upper))))
    dual_gamma = tf.nn.relu(dual_gamma+replaceinf(replacenan(lagrangian/(-best_p))))
  cost1=tf.math.multiply(best_p*100, tf_c1)
  cost2=tf.math.multiply(K.square(best_p*100), tf_c2)
  cost = tf.reduce_sum(cost1+cost2)