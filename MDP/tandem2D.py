import numpy as np

rho = 0.95

lambda1 = 1 * rho 

mu1 = 1 
mu2 = 1 
M = 1000
Delta1 = 1.0            
epsLD = 0.1

r = 0.01
outputname = 'M'+str(M)+'rho'+str(rho)+'r'+str(r)+'Delta'+str(Delta1)+'Tandem' 
print('rho=',rho,'Delta=',Delta1,'r=',r,'M=',M)
rec = []
c = np.array([1.0 ,1.0 + Delta1]);
if r == 0:
    V = np.zeros([M+1,M+1])
else:
    V = np.ones([M+1,M+1]) / r

xlist = np.arange(M + 1)
Y, X= np.meshgrid(xlist, xlist)
action = np.zeros([M+1,M+1])
eps = 2
t = 0
while eps > epsLD:
    
    temp = c[0] * X + c[1] * Y 

    temp[0:M-1,:] = temp[0:M-1,:] + lambda1 * V[1:M,:]

    temp[:,1:M] = temp[:,1:M] + mu2 * V[:,0:M-1] 
    temp[:,0] = temp[:,0] + mu2 * V[:,0] 
    
    
    temp[0,0:M] = temp[0,0:M] + mu1 * V[0,0:M]
    
   
    U1 = mu1 * V[0:M-1,1:M]
    U2 = mu1 * V[1:M,0:M-1]
    
    temp1 = temp.copy()

    temp[1:M,0:M-1] = temp[1:M,0:M-1] + np.minimum(U1,U2)
    G =   (r + lambda1  + mu1 + mu2)
   


    newV = temp / G
    
    rec.append(newV[0,0])
    recV = newV[0,0]
    if r == 0:
        newV = newV - newV[0,0]

    eps = np.linalg.norm(newV[0:M//2,0:M//2]-V[0:M//2,0:M//2])
    V=newV
    
    
    preaction = action.copy()
    action= 1 * (U1 < U2)
    action[0,0:M] = 0
    if t % 100 == 0:
        print("0 0: ",recV)
        print('idle times: ', sum(sum(action[1:200,0:200] == 0)))

        print("action difference: ",round(np.linalg.norm(action[0:M//2,0:M//2]-preaction[0:M//2,0:M//2])** 2))
        
        
        print('Iteration: ', t)
        print(eps)
        
    t = t + 1
    if t % 1000 == 0:
        np.save('data/V' + outputname, V)
        np.save('data/action'+ outputname, action)
        np.save('data/rec'+ outputname, rec)
    
    
    if r == 0:

        V[M,:] = (c[0] * X[M,:] + c[1] * Y[M,:]) ** 2  /(1-rho)
        V[:,M] =  (c[0] * X[:,M] + c[1] * Y[:,M]) ** 2  /(1-rho)
        V[(M-1),:] = V[M,:]
        V[:,(M-1)] = V[:,M]
      
    else:

        V[M,:] = M * (c[0] + c[1])/r
        V[:,M] = M * (c[0] + c[1])/r
        V[(M-1),:] = (M-1) * (c[0] + c[1])/r
        V[:,(M-1)] = (M-1) * (c[0] + c[1])/r

       

np.save('data/V' + outputname, V)
np.save('data/action'+ outputname, action)
np.save('data/rec'+ outputname, rec)


    