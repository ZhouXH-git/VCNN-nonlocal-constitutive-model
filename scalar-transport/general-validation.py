import foam_utilities as foam 
import os, shutil
import subprocess
import multiprocessing

import numpy as np
import pandas as pd
import random
np.random.seed(0)

import torch
import torch.nn as nn

# geometry parameter
aw=1

# number of datapoints used as input in a stencil
# stencil_size = full

# coefficient in equation
epsilon = 0.2
diff = 0.1
zeta = 3

# number of cells in x and y direction
ncellx = round(aw * 200)
ncelly = int(200)
ncell = int(ncellx * ncelly)

# geometry
scale = 0.0357143
Lx = (252 - 54 * (1-aw) * 2) * scale

# directory location
caseBase = '/Users/xuhuizhou/working/periodicHill-auto/transport_'
caseDir = caseBase + str(aw)
timeDir1 = '10'
timeDir2 = '0'

# read data from directory
ufile = os.path.join(caseDir, timeDir1, 'U') # velocity
U = foam.read_vector_field(ufile)

cfile = os.path.join(caseDir, timeDir1, 'C') # cell centers
Cc = foam.read_vector_field(cfile)

strainfile = os.path.join(caseDir, timeDir1, 'S')  # strain rate magnitude
strain = foam.read_scalar_field(strainfile)

cellVolumefile = os.path.join(caseDir, timeDir1, 'V')  # cellVolume
cellVolume = foam.read_scalar_field(cellVolumefile)

wDfile = os.path.join(caseDir, timeDir2, 'wallDistance') # wall distance 
wD = foam.read_scalar_field(wDfile)

boundary = np.zeros([ncell,1]) # boundary information (Yes: 1; No: 0)
boundary[0:ncellx,:] = np.ones([ncellx,1])
boundary[(ncell-ncellx):ncell,:] = np.ones([ncellx,1])

tfile = os.path.join(caseDir, timeDir1, 'T') # temperature
T = foam.read_scalar_field(tfile)

# generate input data
data = np.zeros([ncell,9])

data[:,0:2] = Cc[:,0:2]
data[:,2:4] = U[:,0:2]
data[:,4:5] = strain.reshape(ncell,1)
data[:,5:6] = wD.reshape(ncell,1)
data[:,6:7] = boundary
data[:,7:8] = T.reshape(ncell,1)
data[:,8:9] = cellVolume.reshape(ncell,1)
df = pd.DataFrame(data,columns=['Cx','Cy','u','v','strain','wD','boundary','T','cellVolume'])

df['U']=list(map(lambda x,y: np.sqrt(x**2+y**2), df['u'], df['v'])) # add column of velocity mag "U"
Umax = df.loc[:,"U"].max()
outlength = np.abs(2*diff*np.log(epsilon)/(Umax-np.sqrt(Umax**2+4*diff*zeta))) # region outside the domain because of periodic hills
dfout1 = df[(df['Cx'] > 0) & (df['Cx'] < outlength)]
dfout2 = df[(df['Cx'] > (Lx - outlength)) & (df['Cx'] < Lx)]
dfout1new = dfout1.copy(deep=True)
dfout1new['Cx'] = dfout1new['Cx'] + Lx
dfout2new = dfout2.copy(deep=True)
dfout2new['Cx'] = dfout2new['Cx'] - Lx
dfdata = df.append(dfout1new,ignore_index=True)
dfdata = dfdata.append(dfout2new,ignore_index=True)

Ly = np.sqrt(diff/zeta)*np.abs(np.log(epsilon)) # distance of diffusion only
# define wall distance function
def wDfunc(wallDistance):

    if wallDistance > 1.5*Ly:
        wallfunc = 1.0
    else:
        wallfunc = wallDistance/(1.5*Ly)
        
    return wallfunc

dfdata['wallinfo']=list(map(lambda x: wDfunc(x), dfdata['wD'])) # add column of wall information (function of wD)
dfdata = dfdata.drop(columns=['wD']) # delete the original wall distance column

# define the nonlocal region from local information (u,v,Cx,Cy)
def ellipse(x,y,u,v,a,b,Cx,Cy): 

    alpha = np.arctan(v/u)
    xp = (x-Cx)*np.cos(alpha)+(y-Cy)*np.sin(alpha)
    yp = (x-Cx)*np.sin(alpha)-(y-Cy)*np.cos(alpha)
    value = xp**2/a**2+yp**2/b**2

    return value

# define the influence of the relative position and velocity of certain point to the center point
def rfinalfunc(Cx,Cy,u,v):
    rmag = np.sqrt(Cx**2+Cy**2)
    rfun = 0.01/(rmag+0.01)
    umag = np.sqrt(u**2+v**2)
    cos = -1*(Cx*u+Cy*v)/(rmag*umag+1e-10)
    rfinal = 0.5*(cos+1.05)*umag*rfun
    return rfinal

# neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,64)

        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,1)
        
        
    def forward(self, X):
        R1 = X           # Batchsize * stencil_size * n_features
        stencil = R1.shape[1]
        R2 = X[:,:,4:11] 
        G1 = self.relu(self.fc1(R2))
        G1 = self.relu(self.fc2(G1))
        G1 = self.fc3(G1)
        G2 = G1[:,:,0:4]
        
        G1t = G1.permute(0,2,1)
        R1t = R1.permute(0,2,1)
        D1 = torch.bmm(G1t,R1)/stencil
        D2 = torch.bmm(R1t,G2)/stencil
        D = torch.bmm(D1,D2)   # D = G1t*R*Rt*G2
        
        Dp = D.reshape(D.shape[0],-1)
        out = self.relu(self.fc4(Dp))
        out = self.fc5(out)
        
        return out

# load the well-trained model
# model = torch.load('/Users/xuhuizhou/working/periodicHill-auto/parametric_study_4_cpu.pt')
model = torch.load('/Users/xuhuizhou/working/periodicHill-auto/parametric_study_stencil-25_cpu.pt')

# get input data with full stencil size and get the prediction results with trained model
data_ind=0
prediction_field = torch.empty(int(ncellx*ncelly), 1)

for row in dfdata.itertuples():
    # define the local elliptical region
    ellx = np.abs(2*diff*np.log(epsilon)/(getattr(row,'U')-np.sqrt(getattr(row,'U')**2+4*diff*zeta)))
    elly = np.sqrt(diff/zeta)*np.abs(np.log(epsilon))
    
    # take points in the region
    df1 = dfdata[ellipse(dfdata['Cx'],dfdata['Cy'],getattr(row,'u'),getattr(row,'v'),ellx,elly,getattr(row,'Cx'),getattr(row,'Cy')) < 1.0]
    
    # calculate the relative coordinate
    df2 = df1.copy(deep=True)
    df2['Cx'] = df2['Cx'] - getattr(row,'Cx')
    df2['Cy'] = df2['Cy'] - getattr(row,'Cy')
    
    # add a new column of relative distance to the centre point
    df3 = df2.copy(deep=True)
    df3['rD']=list(map(lambda x,y: np.sqrt(x**2+y**2), df3['Cx'], df3['Cy']))
    
    df4 = df3.drop(columns=['T']) # drop the label column
    df4['rfianl']=list(map(lambda x,y,u,v: rfinalfunc(x,y,u,v),df4['Cx'],df4['Cy'],df4['u'],df4['v'])) 
    
    # turn the dataFrame to array to Tensor with dimension [1,:,:]
    input_array = df4.to_numpy()
    input_tensor = torch.tensor(input_array).to(dtype=torch.float)
    input_tensor = input_tensor.unsqueeze(0) # add another dimension (n=1) in the 0 position
    
    input_tensor[:,:,0] = input_tensor[:,:,0]/(input_tensor[:,:,9]+1e-5)
    input_tensor[:,:,1] = input_tensor[:,:,1]/(input_tensor[:,:,9]+1e-5)
    input_tensor[:,:,9] = 0.01/(0.01+input_tensor[:,:,9])
    
    # pass the input_tensor to the trained neural network
    T_output = model(input_tensor)
    T_output = T_output.reshape(1)
    
    # store the prediction results
    prediction_field[data_ind] = T_output
    data_ind+=1
    
    if getattr(row,'Index')==(ncellx*ncelly-1):  
        break

T = T.reshape(int(ncellx*ncelly),1) # ground truth of temperature
prediction_field= prediction_field.detach().numpy() # prediction of temperature
np.save('/Users/xuhuizhou/working/periodicHill-auto/prediction-stencil_25.npy',prediction_field)

# get the validation error
err_field = np.sqrt(np.sum(np.abs(prediction_field-T)**2))
true_field = np.sqrt(np.sum(np.abs(T)**2))
validation_error = err_field/true_field

print('when aw = {}, the validation error is {};'.format(aw,validation_error))

