import foam_utilities as foam 
import os, shutil
import subprocess
import multiprocessing

import numpy as np
import pandas as pd
import random
np.random.seed(0)

# geometry parameter
aw=1

# number of datapoints used as input in a stencil
stencil_size = 60

# coefficient in equation
epsilon = 0.2
diff = 0.02
zeta = 3

# number of cells in x and y direction
ncellx = round(aw * 200)
ncelly = int(200)
ncell = int(ncellx * ncelly)

# geometry
scale = 0.0357143
Lx = (252 - 54 * (1-aw) * 2) * scale

# directory location
caseBase = '/Users/xuhuizhou/working/new_peri_auto/transport_'
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

# get nonlocal datapoints
data_ind=0
dataX_current=np.zeros([ncellx*ncelly, stencil_size, 11])
dataY_current=np.zeros([ncellx*ncelly, 1])

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
    
    # rearrange the points (relative distance: from small to large)
    df4 = df3.copy(deep=True)
    df4.sort_values("rD",inplace=True)
    
    # uniform sampling from the points in the local elliptical region
    if len(df4) == stencil_size:
        df5 = df4
    elif len(df4) > stencil_size:
        df5 = df4.sample(n=stencil_size,random_state=1,replace=False)
    else:
        df4out = df4.sample(n=(stencil_size-len(df4)),random_state=1,replace=True)
        df5 = df4.append(df4out,ignore_index=True)

    # normalize the cell volume
    df6 = df5.copy(deep=True)
    V_max = df6.loc[:,"cellVolume"].max()
    df6['cellVolume'] = df6['cellVolume'] / V_max
    
    df7 = df6.drop(columns=['T']) # drop the label column
    df7['rfianl']=list(map(lambda x,y,u,v: rfinalfunc(x,y,u,v),df7['Cx'],df7['Cy'],df7['u'],df7['v'])) 
    
    dataX_current[data_ind,:,:]=df7.to_numpy()
    dataY_current[data_ind,0]=getattr(row,'T')
    data_ind+=1
        
    if getattr(row,'Index')==(ncellx*ncelly-1):  
        break

# save the data
np.save('/Users/xuhuizhou/working/periodicHill-auto/new_raw_data/Input_{}.npy'.format(aw), dataX_current)
np.save('/Users/xuhuizhou/working/periodicHill-auto/new_raw_data/Output_{}.npy'.format(aw), dataY_current)




