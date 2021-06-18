import numpy as np
import torch.nn.functional as F
import torch
from loss_deepsupvision import MulticlassDice
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(model, criterion, optimizer, train_dataloaders):  
    dt_size = len(train_dataloaders.dataset)
    model.train()
    epoch_loss = 0   
    epoch_dice = 0   
    epoch_classdice=[0,0,0]
    step = 0
    for img0,img1,img2,y in train_dataloaders:
        step += 1
        input0 = img0.cuda()
        input1 = img1.cuda()
        input2 = img2.cuda()
        target = y.cuda()
        optimizer.zero_grad()
        outputs4,outputs3,outputs2,outputs1,outputs = model(input0,input1,input2)
        loss= criterion(outputs4,outputs3,outputs2,outputs1,target.long(),target.long(),target.long(),target.long())
        current_batchsize = outputs1.size()[0]
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*current_batchsize
        print("%d/%d,train_loss:%0.8f" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item()))
        output_s = F.softmax(outputs,dim=1) 
        output_s = output_s.argmax(dim=1)
        output_s = output_s.unsqueeze(dim=1)
        calu = MulticlassDice(num_classes=3)
        dicee,eachclassdice = calu(output_s,target.long())
        epoch_dice += dicee.item()*current_batchsize
        for i in range(len(epoch_classdice)):
            epoch_classdice[i] += eachclassdice[i].item()*current_batchsize
    epochmean = epoch_loss/dt_size
    epochmeandice = epoch_dice/dt_size
    epochclassmeandice = [i/dt_size for i in epoch_classdice]
    return epochmean,epochclassmeandice[-1]

def val_model(model, criterion,  val_dataloaders): 
    dt_size_val = len(val_dataloaders.dataset)
    model.eval()
    epoch_loss_val = 0
    epoch_dice_val = 0   
    epoch_classdice_val=[0,0,0]
    step_val = 0
    for img0,img1,img2,y in val_dataloaders:  
        step_val += 1
        input0 = img0.cuda()
        input1 = img1.cuda()
        input2 = img2.cuda()
        target = y.cuda()
        outputs4,outputs3,outputs2,outputs1,outputs = model(input0,input1,input2)
        loss = criterion(outputs4,outputs3,outputs2,outputs1,target.long(),target.long(),target.long(),target.long()) #注意注意！
        current_batchsize = outputs1.size()[0]
        epoch_loss_val += loss.item()*current_batchsize 
        output_s = F.softmax(outputs,dim=1)
        output_s = output_s.argmax(dim=1)
        output_s = output_s.unsqueeze(dim=1)
        calu = MulticlassDice(num_classes=3)
        dicee,eachclassdice = calu(output_s,target.long())
        epoch_dice_val += dicee.item()*current_batchsize
        for i in range(len(epoch_classdice_val)):
            epoch_classdice_val[i] += eachclassdice[i].item()*current_batchsize
    epochmean_val = epoch_loss_val/dt_size_val
    epochmeandice_val = epoch_dice_val/dt_size_val
    epochclassmeandice_val  =[i/dt_size_val for i in epoch_classdice_val]
    return epochmean_val,epochclassmeandice_val[-1]