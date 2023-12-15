import numpy as np

rho = 0.95

lambda1 = 1
lambda2 = 1 * rho 

mu1 = 2 
mu2 = 2 
mu3 = 1 
M = 300
Delta1 = 0.0             
Delta2 = 0.5
epsLD = 0.01

r = 0.01
outputname = 'M'+str(M)+'rho'+str(rho)+'r'+str(r)+'Delta'+str(Delta1)+'_'+str(Delta2) +'eps' + str(epsLD)
print(outputname)
print('rho=',rho,'Delta=',Delta1,Delta2,'r=',r,'M=',M)
rec = []
c = np.array([1.0 + Delta1,1.0,1.0 + Delta2]);
if r == 0:
    V = np.zeros([M+1,M+1,M+1])
else:
    V = np.ones([M+1,M+1,M+1]) / r

xlist = np.arange(M + 1)
Y, X, Z = np.meshgrid(xlist, xlist,xlist)
action = np.zeros([M+1,M+1,M+1])
eps = 2
t = 0
while eps > epsLD:
    temp = c[0] * X + c[1] * Y + c[2] *Z

    temp[0:M-1,:,:] = temp[0:M-1,:,:] + lambda1 * V[1:M,:,:]
    

    temp[:,0:M-1,:] = temp[:,0:M-1,:] + lambda2 * V[:,1:M,:]

    
    temp[:,:,1:M] = temp[:,:,1:M] + mu3 * V[:,:,0:M-1] 
    
    
    temp3 = temp.copy()
    temp[0,1:M,0:M-1] =temp[0,1:M,0:M-1]  + mu2 * V[0,0:M-1,1:M]
    temp[1:M,0,:] =temp[1:M,0,:]  + mu1 * V[0:M-1,0,:]
   
    U1 = mu1 * V[0:M-1,1:M,0:M-1]
    U2 = mu2 * V[1:M,0:M-1,1:M]
    temp1 = temp.copy()
    temp2 = temp.copy()
    temp1[1:M,1:M,0:M-1] = temp[1:M,1:M,0:M-1] + U1
    temp2[1:M,1:M,0:M-1] = temp[1:M,1:M,0:M-1] + U2


    G1 =   (r + lambda1 + lambda2 + mu1 * (X>=1) +  mu2 * (Y>=1) * (X==0) + mu3 * (Z >=1))
    G2 =   (r + lambda1 + lambda2 + mu1 * (X>=1) * (Y==0) +  mu2 * (Y>=1) + mu3 * (Z >=1))
    G3 =   (r + lambda1 + lambda2 + mu3 * (Z >= 1))
    V1 = temp1/G1
    V2 = temp2/G2
    V3 = temp3/G3



    newV = np.minimum(V1,V2)
    Vprime = newV.copy()
    newV[0,1:M,0:M-1] = np.minimum(Vprime[0,1:M,0:M-1],V3[0,1:M,0:M-1])
    rec.append(newV[0,0,0])
    print("0 0 0: ",newV[0,0,0])
    if r == 0:
        newV = newV - newV[0,0,0]

    eps = np.linalg.norm(newV[0:M//2,0:M//2,0:M//2]-V[0:M//2,0:M//2,0:M//2])
    V=newV
    
    
    preaction = action.copy()
    action= 1 * (V1 < V2)
    action[0,1:M,0:M-1] = 2 * (V3[0,1:M,0:M-1] < Vprime[0,1:M,0:M-1])
    print('idle times: ', sum(sum((V3[0,1:M,0:M-1] < Vprime[0,1:M,0:M-1]))))
    action[1:M,0,0:M-1] = 1
    action[0,0,0:M-1] = 2
    print("action difference: ",round(np.linalg.norm(action[0:M//2,0:M//2,0:M//2]-preaction[0:M//2,0:M//2,0:M//2])** 2))
    
    
    print('Iteration: ', t)
    t = t + 1
    if t % 1000 == 0:
        np.save('data/V' + outputname, V)
        np.save('data/action'+ outputname, action)
        np.save('data/rec'+ outputname, rec)
        
    
    print(eps)
    if r == 0:
        V[M,:,:] = V[M-1,:,:] 
        V[:,M,:] = V[:,M-1,:] 
        V[:,:,M] = V[:,:,M-1] 
        V[M,:,:] = (M*(c[0]  + c[1]  + c[2])) ** 2  /(1-rho)
        V[:,M,:] = (M*(c[0]  + c[1]  + c[2])) ** 2  /(1-rho)
        V[:,:,M] = (M*(c[0]  + c[1]  + c[2])) ** 2  /(1-rho)
        V[M-1,:,:] = V[M,:,:]
        V[:,M-1,:] = V[:,M,:] 
        V[:,:,M-1] = V[:,:,M]
    else:
        V[M,:,:] = V[M-1,:,:] + c[0]/r
        V[:,M,:] = V[:,M-1,:] + c[1]/r
        V[:,:,M] = V[:,:,M-1] + c[2]/r
        V[M,:,:] = M*(c[0]  + c[1]  + c[2])/r
        V[:,M,:] = M*(c[0]  + c[1]  + c[2])/r
        V[:,:,M] = M*(c[0]  + c[1]  + c[2])/r
        V[M-1,:,:] = M*(c[0]  + c[1]  + c[2])/r
        V[:,M-1,:] = M*(c[0]  + c[1]  + c[2])/r
        V[:,:,M-1] = M*(c[0]  + c[1]  + c[2])/r
    

np.save('data/V' + outputname, V)
np.save('data/action'+ outputname, action)
np.save('data/rec'+ outputname, rec)



    