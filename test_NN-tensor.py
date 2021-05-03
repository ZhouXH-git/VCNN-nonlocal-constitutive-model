import pytest
import numpy as np
import random

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(4100)

# define the NN architecture for Reynolds stress tensor based on point clouds
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7,64)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,64)
        
    def forward(self, X):
        R1 = X           # Batchsize * stencil_size * n_features => B*150*11
        stencil = R1.shape[1]
        R2 = X[:,:,6:13]
        G1 = self.relu(self.fc1(R2))
        G2 = G1[:,:,0:4]
        
        G1t = G1.permute(0,2,1)
        R1t = R1.permute(0,2,1)
        D1 = torch.bmm(G1t,R1)/stencil
        D2 = torch.bmm(R1t,G2)/stencil
        D = torch.bmm(D1,D2)   # D = G1t*R*Rt*G2
        
        xtilde = D1[:,:,0:3]
        xtildet = xtilde.permute(0,2,1)
        
        Dp = D.reshape(D.shape[0],-1)
        out = self.relu(self.fc2(Dp))
        out = self.relu(self.fc3(out))
        E = self.fc4(out)
        E = torch.diag_embed(E)
        
        out = torch.bmm(xtildet,E)
        finalout = torch.bmm(out,xtilde)
        
        return finalout


# define the rotation process
def rotate(input_matrix,rota_matrix):
    xy = input_matrix[:,:,0:3]
    uv = input_matrix[:,:,3:6]
    xyt = xy.permute(0,2,1)
    uvt = uv.permute(0,2,1)
    newxyt = torch.bmm(rota_matrix,xyt)
    newuvt = torch.bmm(rota_matrix,uvt)
    newxy = newxyt.permute(0,2,1)
    newuv = newuvt.permute(0,2,1)
    input_matrix[:,:,0:3] = newxy
    input_matrix[:,:,3:6] = newuv
    newinput_matrix = input_matrix

    return newinput_matrix

B = 16 # batch size
N = 150 # number of points

# give random rotation matrix
rota_matrix = torch.rand(B,3,3)
# angle = random.uniform(0,2*np.pi)
angle = np.pi/2

rota_array = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rota_matrix[:,0:3,0:3] = torch.tensor(rota_array)
rota_matrix_t = rota_matrix.permute(0,2,1)

# unit testing
def test_NN():
    model = Net()
    input1 = torch.rand(B, N, 13)
    output1 = model(input1)
    input2 = rotate(input1,rota_matrix)
    output2 = model(input2)
    # output2_rota = R*output2*Rt
    output2_rota = torch.bmm(rota_matrix_t,output2)
    output2_rota = torch.bmm(output2_rota,rota_matrix)

    assert torch.norm(output1 - output2_rota) < 1e-7   

