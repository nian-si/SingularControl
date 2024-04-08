import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from tqdm import tqdm
import sys
import time


class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        
        
        self.dim = eqn_config.dim
        self.gamma = eqn_config.discount
        self.x0 = eqn_config.x0
        self.var = 1
        if "var" in eqn_config:
            self.var = eqn_config.var
        
        self.a = eqn_config.a
        

    
   


    def gen_samples(self, num_sample,T,N, Total_iterations,  simulation_method):
        delta_t = T / N 
        R = self.R
        sqrt_delta_t = np.sqrt(delta_t)
        dw_sample = np.random.normal(size=[num_sample, self.dim, N * Total_iterations])# * sqrt_delta_t
        dw_sigma_t = np.zeros(dw_sample.shape)
        x0 = np.ones(self.dim) * self.x0;
        
        y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])

        y_sample_array = []
        

        for i in range(N * Total_iterations):
            if i % 1000 == 0:
                print(np.mean(y_i,0))
                print("Generate Samples:",i)

            
            if i % N == 0:

                if simulation_method == "fixed":
                    y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])
                elif simulation_method == "uniform":
                    y_i = np.random.uniform (size = (num_sample, len(x0))) * 2
                y_sample_array.append(y_i)

            dw_sigma_t[:,:,i] = self.diffusion(dw_sample[:, :, i]) * sqrt_delta_t
           

            delta_x = self.drift() * delta_t + dw_sigma_t[:,:,i]

            x_i = y_i + delta_x
            y_i = self.Skorokod(x_i, R)
            y_sample_array.append(y_i)
            


        return np.stack(y_sample_array, axis = 2), dw_sigma_t
        
    def Skorokod(self,X_full, R):
        
        eps = 1e-8
        num_samples = X_full.shape[0]
        Y_ret = np.zeros(X_full.shape)
        for i in range(num_samples):
            X = X_full[i,:]
            Y = X
            while sum (Y < -eps) >0:
                base = (Y < eps)
                R_b = R[base][:,base]

                if len(R_b) > 0:
                    L_b = - np.linalg.solve(R_b, X[base])
                    Y = X + R[:, base] @ L_b
            Y_ret[i,:] = Y
        return Y_ret
 

    def w_tf(self, x):
        """Running cost in control problems."""
        raise NotImplementedError

        
    def V_true(self, x):
        """True value function"""
        raise NotImplementedError

    def V_grad_true(self, x):
        raise NotImplementedError

    def c_true(self):
        """True cost"""
        raise NotImplementedError
    
        
    def sigma(self): #num_sample x dim x dim_w
        """diffusion coefficient"""
        raise NotImplementedError
        
    def drift(self):
        """drift in the SDE"""
        raise NotImplementedError
    
    def diffusion(self,  dw):
        """diffusion in the SDE"""
        raise NotImplementedError






class singularControl(Equation): 
    def __init__(self, eqn_config):
        super(singularControl, self).__init__(eqn_config)
        self.name = 'singularControl'
        self.R = np.identity(self.dim)
        
        self.Delta1 = eqn_config.Delta1
        self.Delta2 = eqn_config.Delta2
        self.Delta = self.Delta1
        self.h1 = tf.constant([-2 * self.Delta2, (1+ self.Delta2)], dtype = tf.float64)
        self.h2 = tf.constant([2 * (1 + self.Delta1), -self.Delta1], dtype = tf.float64)
        self.mu = np.repeat(eqn_config.mu,self.dim) 
        print("Delta 1: %.2f, Delta 2: %.2f" % (self.Delta1, self.Delta2))
    
        
    def w_tf(self, x, grad_t, a_lowbound  = 0): #num_sample * 1
        condition = tf.greater_equal(x[:, 1], 2 * x[:, 0])
        
        
        
        w = tf.where(condition, tf.linalg.matvec(x,self.h1), tf.linalg.matvec(x,self.h2) )
        w = w- tf.linalg.matvec(grad_t,self.mu + tf.constant([0.5,1.0],dtype = tf.float64))

        w = tf.reshape(w, [-1,1])
        max_zero_grad = tf.math.maximum(tf.cast(0., tf.float64) , grad_t)
        min_zero_grad = tf.math.minimum(tf.cast(0., tf.float64) , grad_t)
        w = w + tf.reduce_sum(max_zero_grad, 1, keepdims=True) * a_lowbound  
        w = w + tf.reduce_sum(min_zero_grad, 1, keepdims=True) * self.a
        zero_grad = tf.math.minimum(tf.cast(0., tf.float64),tf.where(condition, grad_t[:,1], grad_t[:,0] ) )
        return w, tf.reduce_sum(tf.square(zero_grad))



    def sigma(self): # x is num_sample x dim, u is num_sample x dim_u, sigma is num_sample x dim x dim_w

        mat1 = np.array([[1,0.5],[0.5,2]])
        return np.linalg.cholesky(mat1)
 
    
    def drift(self):
        return self.mu
    
    def diffusion(self,  dw): #sigma num_sample x dim x dim_w, dw is num_sample x dim_w
        return np.dot(self.sigma(), dw.transpose()).transpose() # num_sample x dim
    
    def const_control_optimal(self):
        pass
    
    def linear_control_optimal(self):
        pass
class bigStep(Equation): 
    def __init__(self, eqn_config):

        super(bigStep, self).__init__(eqn_config)
        self.name = 'bigStep'
        self.R = np.identity(self.dim)
        
        self.Delta1 = eqn_config.Delta1
        self.Delta2 = eqn_config.Delta2

        self.v1 =  eqn_config.v1
        self.v2 =  eqn_config.v2
        self.v3 =  eqn_config.v3

        self.h1 = tf.constant([self.Delta2,self.Delta1], dtype = tf.float64) 
        self.h2 = tf.constant([self.Delta1,self.Delta2], dtype = tf.float64) 
        self.mu = np.repeat(eqn_config.mu,self.dim ) 
        self.mu = self.mu;
        self.realmu = np.repeat(-5.0,self.dim )  
        self.a = self.a  
        
    
        
    def w_tf(self, x, grad_t, a_lowbound  = 0): #num_sample * 1
        condition = tf.greater_equal(x[:, 1],  x[:, 0])
        #w = tf.linalg.matvec(x,self.h)
        w = tf.where(condition, tf.linalg.matvec(x,self.h1), tf.linalg.matvec(x,self.h2) )
        w = w - tf.linalg.matvec(grad_t,self.mu- self.realmu)

        w = tf.reshape(w, [-1,1])
        
        
        max_zero_grad1 = tf.math.maximum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([0.5,1],tf.float64)) - self.v1 )
        max_zero_grad2 = tf.math.maximum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([1.5,1.5],tf.float64)) - self.v2 )
        max_zero_grad3 = tf.math.maximum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([1,1.5],tf.float64)) - self.v3 )
        max_zero_grad = max_zero_grad3 + max_zero_grad1 + max_zero_grad2
       

        min_zero_grad1 = tf.math.minimum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([0.5,1],tf.float64)) - self.v1 )
        min_zero_grad2 = tf.math.minimum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([1.5,1.5],tf.float64)) - self.v2 )
        min_zero_grad3 = tf.math.minimum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([1,1.5],tf.float64)) - self.v3 )
        min_zero_grad = min_zero_grad3 + min_zero_grad1 + min_zero_grad2

        
        w = w - tf.reshape(max_zero_grad, [-1,1]) * (self.a - a_lowbound) 
        
        min_zero_gradG12 = tf.math.minimum(tf.cast(0., tf.float64) , grad_t)
        min_zero_gradG23 = tf.math.minimum(tf.cast(0., tf.float64) , tf.linalg.matvec(grad_t,tf.cast([2,3],tf.float64)) )
        w = w + tf.reduce_sum(min_zero_gradG12, 1, keepdims=True) * (self.a - a_lowbound) 
        w = w + tf.reshape(min_zero_gradG23, [-1,1]) * (self.a - a_lowbound) 
        
        zero_grad = tf.math.minimum(tf.cast(0., tf.float64),tf.where(condition, grad_t[:,1], grad_t[:,0] ) )
        zero_grad0 = tf.math.minimum(tf.cast(0., tf.float64),grad_t[:,0]  )
        zero_grad1 = tf.math.minimum(tf.cast(0., tf.float64),grad_t[:,1]  )
        return w, tf.reduce_sum(tf.square(zero_grad0)) + tf.reduce_sum(tf.square(zero_grad1))
        #return w, tf.reduce_sum(tf.square(zero_grad))
        



    def sigma(self): # x is num_sample x dim, u is num_sample x dim_u, sigma is num_sample x dim x dim_w

        mat1 = np.array([[50,54],[54,69]])
        mat1 = mat1 
        return np.linalg.cholesky(mat1)
 
    def gen_samples(self, num_sample,T, N, Total_iterations,  simulation_method):
       
        delta_t = T /N  
        R = self.R
        sqrt_delta_t = np.sqrt(delta_t)
        dw_sample = np.random.normal(size=[num_sample, self.dim, N * Total_iterations])# * sqrt_delta_t
        dw_sigma_t = np.zeros(dw_sample.shape)
        x0 = np.ones(self.dim) * self.x0;
        
        y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])

        y_sample_array = []
        

        for i in range(N * Total_iterations):
            if i % 1000 == 0:
                
                print("Generate Samples:",i)

            
            if i % N == 0:

                if simulation_method == "fixed":
                    y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])
                elif simulation_method == "uniform":
                    y_i = np.random.uniform (size = (num_sample, len(x0))) * 2
                y_sample_array.append(y_i)

            dw_sigma_t[:,:,i] = self.diffusion(dw_sample[:, :, i]) * sqrt_delta_t
           

            delta_x = self.drift() * delta_t + dw_sigma_t[:,:,i]

            x_i = y_i + delta_x
            y_i = np.zeros(x_i.shape)
            
            y_i[:,0] = np.maximum(0,x_i[:,0])
            y_i[:,1] = np.maximum(x_i[:,1],y_i[:,0] * 0.75)
            

            y_sample_array.append(y_i)
            


        return np.stack(y_sample_array, axis = 2), dw_sigma_t
    def drift(self):
        return self.mu
    
    def diffusion(self,  dw): #sigma num_sample x dim x dim_w, dw is num_sample x dim_w
        return np.dot(self.sigma(), dw.transpose()).transpose() # num_sample x dim
    
    def const_control_optimal(self):
        pass
    
    def linear_control_optimal(self):
        pass

class tandem(Equation): 
    def __init__(self, eqn_config):
        super(tandem, self).__init__(eqn_config)
        self.name = 'tandem'
        self.R = np.identity(self.dim)
        for i in np.arange(self.dim - 1):

            self.R[i + 1, i] = eqn_config.R;
        self.Delta = eqn_config.Delta
      
        self.h = tf.constant(eqn_config.h, dtype = tf.float64)
        self.mu = np.zeros(self.dim)
        self.mu[0] = eqn_config.mu
        
        self.realmu = np.zeros(self.dim)
        self.realmu[0] = eqn_config.realmu
        
        print("Real Mu -", self.realmu)
        print(self.h)

    
        
    def w_tf(self, x, grad_t, a_lowbound  = 0): #num_sample * 1

        w =  tf.linalg.matvec(x,self.h)
        w = w- tf.linalg.matvec(grad_t,self.mu + self.realmu)

        w = tf.reshape(w, [-1,1])
        for i in np.arange(self.dim - 1):
            max_zero_grad = tf.math.maximum(tf.cast(0., tf.float64) , grad_t[:,i] - grad_t[:,i + 1] )
            min_zero_grad = tf.math.minimum(tf.cast(0., tf.float64) , grad_t[:,i] - grad_t[:,i + 1] )
            #w = w + tf.reshape(max_zero_grad, [-1,1]) * a_lowbound  
            w = w + tf.reshape(min_zero_grad, [-1,1]) *  (self.a - a_lowbound)
        

        #zero_grad = tf.math.minimum(tf.cast(0., tf.float64),grad_t ) 
        return w, 0
        
        


    def sigma(self): # x is num_sample x dim, u is num_sample x dim_u, sigma is num_sample x dim x dim_w
        mat1 = np.identity(self.dim) * 2.0
        for i in np.arange(self.dim - 1):
            mat1[i,i+1] = -1.0
            mat1[i+1,i] = -1.0

        return np.linalg.cholesky(mat1)
 
    
    def drift(self):
        return self.mu
    
    def diffusion(self,  dw): #sigma num_sample x dim x dim_w, dw is num_sample x dim_w
        return np.dot(self.sigma(), dw.transpose()).transpose() # num_sample x dim
    
    def const_control_optimal(self):
        pass
    
    def linear_control_optimal(self):
        pass

