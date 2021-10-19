import numpy as np
import torch
import torch.nn as nn

# training list and validation list
aw_train_list = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
aw_valid_list = [0.5]
stencil_size = 150

# import training data (Cx,Cy,u,v,strain,bouninfo,cellVolume,|U|,wallinfo,rD,rfinal)
train_num = 0
for aw in aw_train_list:
    
    dataX_train_current = np.load('/Users/xuhuizhou/working/periodicHill-auto/new_raw_data/Input_{}.npy'.format(aw))
    dataY_train_current = np.load('/Users/xuhuizhou/working/periodicHill-auto/new_raw_data/Output_{}.npy'.format(aw))
    
    dataX_train_current=torch.tensor(dataX_train_current).to(dtype=torch.float)
    dataY_train_current=torch.tensor(dataY_train_current).to(dtype=torch.float)
    
    if train_num < 1:
        dataX_train=dataX_train_current
        dataY_train=dataY_train_current
    else:
        dataX_train=torch.cat((dataX_train,dataX_train_current))
        dataY_train=torch.cat((dataY_train,dataY_train_current))
    
    train_num+=1

# import validation data (Cx,Cy,u,v,strain,bouninfo,cellVolume,|U|,wallinfo,rD,rfinal)
valid_num = 0
for aw in aw_valid_list:
    
    dataX_valid_current = np.load('/Users/xuhuizhou/working/periodicHill-auto/new_raw_data/Input_{}.npy'.format(aw))
    dataY_valid_current = np.load('/Users/xuhuizhou/working/periodicHill-auto/new_raw_data/Output_{}.npy'.format(aw))
    
    dataX_valid_current=torch.tensor(dataX_valid_current).to(dtype=torch.float)
    dataY_valid_current=torch.tensor(dataY_valid_current).to(dtype=torch.float)
    
    if valid_num < 1:
        dataX_valid=dataX_valid_current
        dataY_valid=dataY_valid_current
    else:
        dataX_valid=torch.cat((dataX_valid,dataX_valid_current))
        dataY_valid=torch.cat((dataY_valid,dataY_valid_current))
    
    valid_num+=1

# modify features (Cx/rD, Cy/rD, and 0.01/(0.01+rD))
dataX_train[:,:,0] = dataX_train[:,:,0]/(dataX_train[:,:,9]+1e-5)
dataX_train[:,:,1] = dataX_train[:,:,1]/(dataX_train[:,:,9]+1e-5)
dataX_train[:,:,9] = 0.01/(0.01+dataX_train[:,:,9])
dataX_valid[:,:,0] = dataX_valid[:,:,0]/(dataX_valid[:,:,9]+1e-5)
dataX_valid[:,:,1] = dataX_valid[:,:,1]/(dataX_valid[:,:,9]+1e-5)
dataX_valid[:,:,9] = 0.01/(0.01+dataX_valid[:,:,9])

print(np.isnan(dataX_train.numpy()).any()) # check if there is NAN in the data (If False, OK!)

print(len(aw_train_list))
print(len(aw_valid_list))

torch.save(dataX_train, '/Users/xuhuizhou/working/periodicHill-auto/training-data/new_dataX_train.pt')
torch.save(dataY_train, '/Users/xuhuizhou/working/periodicHill-auto/training-data/new_dataY_train.pt')
torch.save(dataX_valid, '/Users/xuhuizhou/working/periodicHill-auto/training-data/new_dataX_valid.pt')
torch.save(dataY_valid, '/Users/xuhuizhou/working/periodicHill-auto/training-data/new_dataY_valid.pt')


