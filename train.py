import torch
import torch.optim as optim
from data import SMPL_DATA 
from model_maxpool import NPT 
from model2 import NeuralPoseTransfer
import utils as utils 
import numpy as np
import time
import pymesh
import MyMesh
import meshio


batch_size=8

dataset = SMPL_DATA(train=True, shuffle_point = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


chkpath = "./saved_model/144_BIN.model"
epoch_checkpoint = 144
checkpoint = torch.load(chkpath)

model=NPT()
model=NeuralPoseTransfer(norm_type='Batch')
model.cuda()
model.apply(utils.weights_init)

model.load_state_dict(checkpoint['model_state_dict'])

lrate=0.00005
optimizer_G = optim.Adam(model.parameters(), lr=lrate)
optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

print("Begin Training NPT model")

for epoch in range(epoch_checkpoint+1, 200):
    
    start=time.time()
    

    total_loss=0
    for j,data in enumerate(dataloader,0):

        optimizer_G.zero_grad()
        
        pose_points, random_sample, gt_points, identity_points, new_face=data

        
        pose_points=pose_points.transpose(2,1)
        pose_points=pose_points.cuda()

        identity_points=identity_points.transpose(2,1)
        identity_points=identity_points.cuda()

        
        gt_points=gt_points.cuda()

        pointsReconstructed = model(pose_points,identity_points)  

        rec_loss =  torch.mean((pointsReconstructed - gt_points)**2)

        edg_loss=0
        
        for i in range(len(random_sample)):

            f=new_face[i].cpu().numpy()
            v=identity_points[i].transpose(0,1).cpu().numpy()
            edg_loss=edg_loss+utils.compute_score(pointsReconstructed[i].unsqueeze(0),f,utils.get_target(v,f,1))
        
        edg_loss=edg_loss/len(random_sample)

        l2_loss=rec_loss
        rec_loss=rec_loss+0.0005*edg_loss
        rec_loss.backward()
        optimizer_G.step()
        total_loss=total_loss+rec_loss 

    
    print('####################################')
    print(epoch)
    print(time.time()-start)
    mean_loss=total_loss/(j+1)
    print('mean_loss',mean_loss.item())
    print('####################################')

    
    if (epoch+1)%5==0:
        save_path='./saved_model/'+str(epoch)+'_BIN.model'
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(), 
          'optimizer_state_dict': optimizer_G.state_dict()},save_path)

