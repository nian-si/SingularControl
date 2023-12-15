import numpy as np
dim = 2;
xi_orignal=np.array([-5.0,-5.0]);
r = 0.1
h = 0.1
M = 400;
Sigma_original = np.array([[50, 54],[54, 69]])
multi = 54 / 50
h2 = h * multi
xi = np.array([-5.0,-5.0 / multi]);
Sigma= np.array([[50, 54 / multi],[54 / multi, 69 / multi ** 2]])
h1 = h

def create_Qh():
    Qh = 0;
    for i in np.arange(dim):
        Qh = Qh + Sigma[i,i]
        for j in np.arange(dim):
            if j != i:
                Qh -= np.abs(Sigma[i,j]) / 2
        Qh += h * np.abs(xi[i])
    return Qh

Qh = create_Qh()
Delta_t = h* h / Qh
print(Qh)
print(Delta_t)
def gen_prob():
    p ={}
    for i in np.arange(dim):
        cur_p = Sigma[i,i] / 2
        for j in np.arange(dim):
            if j != i:
                cur_p -= np.abs(Sigma[i,j]) / 2
                cur_idx = [0] * dim
                
                if Sigma[i,j] > 0:
                    cur_idx[i] = 1;
                    cur_idx[j] = 1;
                    cur_idx_p = tuple(cur_idx)
                    p[cur_idx_p] = Sigma[i,j] / Qh / 2
                    cur_idx[i] = -1;
                    cur_idx[j] = -1;
                    cur_idx_p = tuple(cur_idx)
                    p[cur_idx_p] = Sigma[i,j] / Qh / 2
                else:
                    cur_idx[i] = -1;
                    cur_idx[j] = 1;
                    cur_idx_p = tuple(cur_idx)
                    p[cur_idx_p] = -Sigma[i,j] / Qh / 2
                    cur_idx[i] = 1;
                    cur_idx[j] = -1;
                    cur_idx_p = tuple(cur_idx)
                    p[cur_idx_p] = -Sigma[i,j] / Qh / 2


        cur_idx = [0] * dim
        cur_idx[i] = 1;
        cur_idx_p = tuple(cur_idx)
        if cur_p + max(0,xi[i]) * h > 0:
            p[cur_idx_p] = (cur_p + max(0,xi[i]) * h) / Qh
        cur_idx[i] = -1;
        cur_idx_n = tuple(cur_idx)
        if cur_p - min(0,xi[i]) * h > 0:
            p[cur_idx_n] = (cur_p - min(0,xi[i]) * h) / Qh


    print(p)
    return p

p = gen_prob()
cur_sum = 0
for direc, prob in p.items():      cur_sum = cur_sum + prob


xlist = np.arange(M )
Y, X = np.meshgrid(xlist, xlist)
action = np.zeros([M,M])
eps = 2
t = 0
V = (X+Y) / r * h  


c=[1,1]
control_cost=[1.65,1.0,2.25] 
control_cost=[2.0,1.0,2.0]
control_cost=[2.0,1.0,1.0]
action_set = {
    (-1,-1.5): control_cost[0] ,
    (-0.5,-1): control_cost[1] ,
    (-1.5,-1.5): control_cost[2] ,
    (1,0):0,
    (0,1):0
}
def process_action(dire, cost):
    if np.abs(dire[0]) >  np.abs(dire[1])  / multi:
        scale = np.abs(dire[0])
        dire = dire / scale

        c = cost / scale
        dict = {}
        key = (np.sign(dire[0]).astype(int), np.sign(dire[1]) .astype(int))
        dict[key] = np.abs(dire[1]) / multi
        key = (np.sign(dire[0]).astype(int), 0) 
        dict[key] = 1 - np.abs(dire[1]) / multi
        return dict, c, h1
    else:
        scale = np.abs(dire[1])
        dire = dire / scale
        c = cost / scale
        dict = {}
        key = (np.sign(dire[0]).astype(int), np.sign(dire[1]).astype(int) )
        dict[key] = np.abs(dire[0]) * multi
        key = (0, np.sign(dire[1]).astype(int))
        dict[key] = 1 - np.abs(dire[0]) * multi
        return dict, c, h2
action_new = []        
for a, c_a in action_set.items():
    action_new.append(process_action(a,c_a))
print(action_new)
discount = np.exp(-Delta_t * r)

print(cur_sum *discount,cur_sum) 
def transition(p, V):
    temp = np.zeros(V.shape)
    for direc, prob in p.items():
        if direc[0] == -1: x_list = np.arange(1,M)
        elif direc[0] == 0: x_list = np.arange(0,M)
        else:  x_list = np.arange(0,M-1)

        if direc[1] == -1: y_list = np.arange(1,M)
        elif direc[1] == 0: y_list = np.arange(0,M)
        else:  y_list = np.arange(0,M-1)

        temp[np.ix_(x_list,y_list)] = temp[np.ix_(x_list,y_list)] + prob * V[np.ix_(x_list + direc[0], y_list + direc[1]) ] 
        #print(temp[0,0],'a')
        if direc[0] == -1:
            temp[0,y_list] = temp[0,y_list] +  prob * V[0,y_list + direc[1]]
            
        
        if direc[1] == -1:
            temp[0,0] = temp[0,0] + prob * V[0,0]
  

            if x_list[0] == 0:
                x_l = x_list[1:]
            else: x_l = x_list
            y_l = np.ceil(x_l * 0.75 / multi).astype(int)
            x_list = x_l[(x_l+direc[0]) * 0.75 > (y_l + direc[1]) * multi ]
            y_list = y_l[(x_l+direc[0]) * 0.75 > (y_l + direc[1]) * multi ]
      

            temp[x_list,y_list] = temp[x_list,y_list] - prob * V[x_list + direc[0],y_list -1] + prob *V[x_list + direc[0],y_list]
        if direc[0] == 1 and direc[1] == 0: #only show up in control
            x_l = x_list
            y_l = np.ceil(x_list * 0.75 / multi).astype(int)
            x_list = x_l[(x_l+direc[0]) * 0.75 > (y_l + direc[1]) * multi ]
            y_list = y_l[(x_l+direc[0]) * 0.75 > (y_l + direc[1]) * multi ]

            
            temp[x_list,y_list] = 1000000

    return temp
print('discount',discount, 'Delta',Delta_t)
res =  []
while eps > 0.01:
    temp = (c[0] * X * h1 + c[1] * Y *  h2) * Delta_t 
    temp = temp + discount * transition(p,V)
    pre_action = action

    action = np.zeros(temp.shape)
    
    action_idx = 0
    for cur_act in action_new:
        action_idx = action_idx + 1
        cur_V = transition(cur_act[0], V) + cur_act[1] *  cur_act[2]
        action[cur_V < temp] = action_idx
        temp[cur_V < temp] = cur_V[cur_V < temp]
    
    

    eps = np.linalg.norm(temp[0:M//2,0:M//2]-V[0:M//2,0:M//2])
    
    V=temp
    action_diff = np.linalg.norm(action[0:M//2,0:M//2]-pre_action[0:M//2,0:M//2]) ** 2

    res.append([t,eps,action_diff,V[0,0]])
    if  t % 500 == 0:
        print(t)
        print(eps)
        
        print("action diff", action_diff)
        print(V)
        print(action)
    if t % 2000 == 0:
        np.save('data/bigstep_V' + 'M'+str(M)+'r'+str(r)+'h'+str(h)+'cost'+str(control_cost), V)
        np.save('data/bigstep_action'+ 'M'+str(M)+'r'+str(r)+'h'+str(h)+'cost'+str(control_cost), action)

        np.savetxt('data/bigstep_res'+ 'M'+str(M)+'r'+str(r)+'h'+str(h)+'cost'+str(control_cost)+'.csv', res, delimiter=",")
    t = t + 1
    V[M-1,:] = M * 100 * c[0]/r  * h
    V[:,M-1] = M * 100 * c[1]/r  * h
    
      

    
    
print(action)
print(V)

np.save('data/bigstep_V' + 'M'+str(M)+'r'+str(r)+'h'+str(h)+'cost'+str(control_cost), V)
np.save('data/bigstep_action'+ 'M'+str(M)+'r'+str(r)+'h'+str(h)+'cost'+str(control_cost), action)
np.savetxt('data/bigstep_res'+ 'M'+str(M)+'r'+str(r)+'h'+str(h)+'cost'+str(control_cost)+'.csv', res, delimiter=",")

    