import argparse
import numpy as np
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataset import LiverDataset
from torch.utils.data import DataLoader
from Network.Models import AttU_Net,NestedUNet
from train_val import train_model, val_model
from utils_pure import softmax,check_dir,plot_acc,remove,create_exp_dir,plot_w,plot,plot_w_acc
from loss_deepsupvision import LossPRSnet_dicece
import os
import glob
import logging
import sys
import time
from torchvision.transforms import transforms


class Mixedpath(nn.Module):
    def __init__(self, n_choices):
        super(Mixedpath, self).__init__()
        self.n_choices = n_choices
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability  是利用softmax把alpha变成概率
        return probs

#Set gpu
gpu_id = 2
torch.cuda.set_device(gpu_id)
device = torch.device('cuda:%d'%(gpu_id))
torch.set_num_threads(2)
#initialization
parser = argparse.ArgumentParser("Need to customize the parameters")
parser.add_argument('--train_datapath', type=str, default='')
parser.add_argument('--val_datapath', type=str, default='')
parser.add_argument('--init_weight', type=str, default='')
args = parser.parse_args()
coeff = np.array([0.5, 0.5, 0.5, 1])
delta = 0.15  #The amount of each change
n_choices = 3   #There are 3 actions, corresponding to addition, subtraction, and unchanged

mix1 = Mixedpath(n_choices).to(device)
mix2 = Mixedpath(n_choices).to(device)
mix3 = Mixedpath(n_choices).to(device)
mix4 = Mixedpath(n_choices).to(device)
sample_num = 10 #采样次数
baseline = None
baseline_decay_weight = 0.99
init_type = 'normal'   #Used for initialization
init_ratio = 1e-3

def architecture_parameters(AP_path_alpha):
    yield AP_path_alpha

# Initialization
for param in mix1.parameters():
    if init_type == 'normal':
        param.data.normal_(0, init_ratio)
    elif init_type == 'uniform':
        param.data.uniform_(-init_ratio, init_ratio)

for param in mix2.parameters():
    if init_type == 'normal':
        param.data.normal_(0, init_ratio)
    elif init_type == 'uniform':
        param.data.uniform_(-init_ratio, init_ratio)

for param in mix3.parameters():
    if init_type == 'normal':
        param.data.normal_(0, init_ratio)
    elif init_type == 'uniform':
        param.data.uniform_(-init_ratio, init_ratio)

for param in mix4.parameters():
    if init_type == 'normal':
        param.data.normal_(0, init_ratio)
    elif init_type == 'uniform':
        param.data.uniform_(-init_ratio, init_ratio)

# Optimizer
arch_optimizer1 = torch.optim.Adam(mix1.parameters(), 1e-3)
arch_optimizer2 = torch.optim.Adam(mix2.parameters(), 1e-3)
arch_optimizer3 = torch.optim.Adam(mix3.parameters(), 1e-3)
arch_optimizer4 = torch.optim.Adam(mix4.parameters(), 1e-3)

#Load data, data transformation
batch_size=16
img_transforms = transforms.Compose([  
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10)
])
normal_transforms = transforms.Compose([ 
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor(),         
    transforms.Normalize((0.5,), (0.5,))
])
normalval_transforms = transforms.Compose([ 
    transforms.ToTensor(),        
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = LiverDataset(args.train_datapath,transform=img_transforms,target_transform=img_transforms,normalization=normal_transforms)    # 这是训练集
print('训练集长度:', len(train_dataset))
train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataset = LiverDataset(args.val_datapath, transform=None,target_transform=None,normalization=normalval_transforms)       # 这是验证集
print('验证集长度:', len(val_dataset))
val_dataloaders = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#Generate folder and copy py file
result_path = '{}-{}'.format('EXP', time.strftime("%Y%m%d-%H%M%S")) #时间EXP-20200915-175950
check_dir(result_path)
log_format = '%(asctime)s %(message)s'  #打印日志时间和信息
logging.basicConfig(stream = sys.stdout, level = logging.INFO,
    format = log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(result_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
create_exp_dir(result_path, scripts_to_save = glob.glob('*.py'))

log_path = os.path.join(result_path, "Log.txt")
logging.info("gpu_id:{}".format(gpu_id))

path=None
dirs='{}/weight'.format(result_path)
train_acc,train_loss,val_acc,val_loss,ww=[],[],[],[],[]
reward_ = []

torch.cuda.synchronize()
start = time.time()

for epoch in range(50):
    obj_term = [0 for i in range(sample_num)] 
    reward_buffer = []
    grad_buffer = []
    coeff_total = np.zeros([sample_num, 4]) 
    best_vacc,best_tacc,best_vloss,best_tloss = 0,0,0,0
    temp = None
    init_weight = path
    if epoch>0:
        checkpoint = torch.load(init_weight,map_location = 'cuda:%d'%(gpu_id))
    #Initialize the model and optimizer every time
    ParallelModel,ParallelOptimizer = [],[]
    for i in range(sample_num):

        model = AttU_Net(in_ch=2, out_ch=3, deep_supervision=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if epoch==0:
            weights=torch.load(args.init_weight,map_location='cuda')
            model.load_state_dict(weights['net'])
        if epoch>0:
            #Initialize the model and optimizer
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        #Add the model and optimizer to the list
        ParallelModel.append(model)        
        ParallelOptimizer.append(optimizer)


    # Model weight copy
    for j in range(sample_num):

        logging.info("epoch:{}, num_sample:{}".format(epoch,j))
        #Generation probability
        probs1 = mix1.probs_over_ops()  #Return three probabilities
        probs2 = mix2.probs_over_ops()
        probs3 = mix3.probs_over_ops()
        probs4 = mix4.probs_over_ops()
        print('probs', probs1.data,probs2.data,probs3.data,probs4.data) #Get tensor value
        #Return the position of the three largest probability values 0 or 1 or 2
        sample1 = torch.multinomial(probs1.data, 1)[0].item()   
        sample2 = torch.multinomial(probs2.data, 1)[0].item()
        sample3 = torch.multinomial(probs3.data, 1)[0].item()
        sample4 = torch.multinomial(probs4.data, 1)[0].item()
        sample = np.array([sample1, sample2, sample3, sample4])
        print('sample', sample-1)
        #Update the real number, softmax gets the coefficient
        coeff_total[j] = coeff + delta*(sample - 1)
        w = softmax(coeff_total[j])
        print(coeff_total[j])
        logging.info("w:[{:.4},{:.4},{:.4},{:.4}]".format(w[0],w[1],w[2],w[3]))


        #Get model and optimizer, use dice_ce loss
        net = ParallelModel[j].cuda()
        optimizer = ParallelOptimizer[j]
        criterion = LossPRSnet_dicece(weights=[w[0],w[1],w[2],w[3],0,1],num_classes=3)

        #Training and prediction
        t_loss,t_dice = train_model(net,criterion,optimizer,train_dataloaders)
        v_loss,v_dice = val_model(net, criterion,val_dataloaders)
        logging.info("t_loss:{:.4}, t_dice:{:.4} || v_loss:{:.4}, v_dice:{:.4}".format(t_loss,t_dice,v_loss,v_dice))

        if v_dice > best_vacc: 

            #Remove the weight model corresponding to the previous non-maximum value to save memory
            if temp!=None:
                remove(temp)

            #Generate path and dictionary
            check_dir(dirs)
            path = '{}/{}_{}_{:.4}.pth.gz'.format(dirs,epoch,j,v_dice)
            state = {'net':net.state_dict(),'optimizer':optimizer.state_dict()}

            #Save data
            torch.save(state, path)
            best_vacc = v_dice
            best_tacc = t_dice
            best_vloss = v_loss
            best_tloss = t_loss
            temp = path
            best_w = w
            max_index = j


        log_prob = torch.log(probs1[sample1])
        obj_term[j] = obj_term[j] + log_prob
        log_prob = torch.log(probs2[sample2])
        obj_term[j] = obj_term[j] + log_prob
        log_prob = torch.log(probs3[sample3])
        obj_term[j] = obj_term[j] + log_prob
        log_prob = torch.log(probs4[sample4])
        obj_term[j] = obj_term[j] + log_prob


        # Custom reward
        reward = (v_dice+0.04)**3
        logging.info("reward:{:.4}".format(reward))
        logging.info("----------------------------------------------")

        #Save the reward
        reward_buffer.append(reward)

        arch_optimizer1.zero_grad()
        arch_optimizer2.zero_grad()
        arch_optimizer3.zero_grad()
        arch_optimizer4.zero_grad()
        loss_term = -obj_term[j]
        loss_term.backward(retain_graph=True)

        #Get the gradient of this sampling probability
        grad_list = []
        for param in mix1.parameters():
            grad_list.append(param.grad.data.clone())

        for param in mix2.parameters():
            grad_list.append(param.grad.data.clone())

        for param in mix3.parameters():
            grad_list.append(param.grad.data.clone())

        for param in mix4.parameters():
            grad_list.append(param.grad.data.clone())

        grad_buffer.append(grad_list)

    # Select the model with the highest reward above and save it as the source for the next model weight copy
    # Choose the coeff_total with the highest reward as the coeff
    coeff = coeff_total[max_index]
    info = [str(epoch).zfill(3), best_vloss,best_vacc]
    logtxt = open(log_path, "a")
    logtxt.write("Epoch: {} | vloss: {:.4f} vacc: {:.4f}\n".format(*info))
    logtxt.write("Coeff: [{:.4f},{:.4f},{:.4f},{:.4f}]\n".format(w[0],w[1],w[2],w[3]))
    print("Epoch: {} | vloss: {:.4f} vacc: {:.4f}\n".format(*info))
    #Save acc and loss, and coefficients
    train_acc.append(best_tacc)
    val_acc.append(best_vacc)
    train_loss.append(best_tloss)
    val_loss.append(best_vloss)
    ww.append(best_w)
    reward_.append(sum(reward_buffer[:]))
    #Drawing
    plot_acc(train_acc,val_acc,result_path,'train_acc','val_acc','acc')
    plot_acc(train_loss,val_loss,result_path,'train_loss','val_loss','loss')
    plot_w(ww,result_path,'coeff','coeff')
    plot(reward_,result_path,'reward','reward')
    plot_w_acc(ww,val_acc,result_path,'coeff-acc','coeff')

    # Start the update
    avg_reward = sum(reward_buffer) / sample_num
    if baseline is None:
        baseline = avg_reward
    else:
        baseline += baseline_decay_weight * (avg_reward - baseline)

    arch_optimizer1.zero_grad()
    arch_optimizer2.zero_grad()
    arch_optimizer3.zero_grad()
    arch_optimizer4.zero_grad()

    for param in mix1.parameters():
        for j in range(sample_num):
            param.grad.data += (reward_buffer[j] - baseline) * grad_buffer[j][0]
        param.grad.data /= sample_num
    arch_optimizer1.step()

    for param in mix2.parameters():
        for j in range(sample_num):
            param.grad.data += (reward_buffer[j] - baseline) * grad_buffer[j][1]
        param.grad.data /= sample_num
    arch_optimizer2.step()

    for param in mix3.parameters():
        for j in range(sample_num):
            param.grad.data += (reward_buffer[j] - baseline) * grad_buffer[j][2]
        param.grad.data /= sample_num
    arch_optimizer3.step()

    for param in mix4.parameters():
        for j in range(sample_num):
            param.grad.data += (reward_buffer[j] - baseline) * grad_buffer[j][3]
        param.grad.data /= sample_num
    arch_optimizer4.step()

torch.cuda.synchronize()
end = time.time()
print('训练时间为：', end-start,'秒')
