import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class get_one_hot(nn.Module):
	def __init__(self):
		super(get_one_hot, self).__init__()
	def	forward(self,label, N):
		size = list(label.size())
		label = label.view(-1)	
		ones = torch.sparse.torch.eye(N)
		ones = ones.index_select(0, label)	
		size.append(N)	
		return ones.view(*size).permute(2,0,1)

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1e-5 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
		intersection = input_flat * target_flat
		loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
		return loss

class MultiDiceLoss(nn.Module):
	"""
	requires one hot encoded target. Applies DiceLoss on each class iteratively.
	requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
	  batch size and C is number of classes
	"""
	def __init__(self, num_classes):
		super(MultiDiceLoss, self).__init__()
		self.num_classes = num_classes
 
	def forward(self, input, target, weights=None):
		input = F.softmax(input,dim=1)  
		N= target.shape[0]  
		each_mapclass=[]
		each_diceloss=[] 
		one_hot=get_one_hot()
		for i in range(N):  
			each_map=target[i,:]  
			each_map=each_map.squeeze(0)
			each_mapp=one_hot(each_map.cpu(),self.num_classes) 
			each_mapp=each_mapp.unsqueeze(0) 
			each_mapclass.append(each_mapp)
		output_target=torch.cat(each_mapclass,dim=0).cuda()
		C = input.shape[1]
		if weights is None:
			weights = torch.ones(C)
		dice = DiceLoss()
		totalLoss = 0
		for i in range(C):
			diceLoss = dice(input[:,i], output_target[:,i])
			if weights is not None:
				diceLoss *= weights[i]
			each_diceloss.append(diceLoss)
			totalLoss += diceLoss
		return totalLoss

#四种loss的加权求和
class LossPRSnet_dicece:
	def __init__(self,num_classes=2,weights=[0.125,0.25,0.5,1,0.2,1]):

		self.criterion0 = MultiDiceLoss(num_classes)
		self.criterion1 = nn.CrossEntropyLoss()
		self.weights = weights 
   
	def __call__(self,outputs0,outputs1,outputs2,outputs3,targets0,targets1,targets2,targets3):
		dc_loss0 = self.criterion0(outputs0, targets0)
		dc_loss1 = self.criterion0(outputs1, targets1)
		dc_loss2 = self.criterion0(outputs2, targets2)
		dc_loss3 = self.criterion0(outputs3, targets3)
		outputs0,outputs1,outputs2,outputs3 = outputs0.squeeze(dim=1),outputs1.squeeze(dim=1),outputs2.squeeze(dim=1),outputs3.squeeze(dim=1)
		targets0,targets1,targets2,targets3 = targets0.squeeze(dim=1),targets1.squeeze(dim=1),targets2.squeeze(dim=1),targets3.squeeze(dim=1),
		ce_loss0 = self.criterion1(outputs0, targets0)
		ce_loss1 = self.criterion1(outputs1, targets1)
		ce_loss2 = self.criterion1(outputs2, targets2)
		ce_loss3 = self.criterion1(outputs3, targets3)
		ce_dc_loss0 = self.weights[4] * ce_loss0 + self.weights[5] * dc_loss0
		ce_dc_loss1 = self.weights[4] * ce_loss1 + self.weights[5] * dc_loss1
		ce_dc_loss2 = self.weights[4] * ce_loss2 + self.weights[5] * dc_loss2
		ce_dc_loss3 = self.weights[4] * ce_loss3 + self.weights[5] * dc_loss3
		criterion = self.weights[0] * ce_dc_loss0 + self.weights[1] * ce_dc_loss1 + self.weights[2] * ce_dc_loss2 + self.weights[3] * ce_dc_loss3
		return criterion

#评估指标dice
#计算每一类的dice
class Dice(nn.Module):
	def __init__(self):
		super(Dice, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1e-5
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
		intersection = input_flat * target_flat
		loss = (2 * intersection.sum(1) + smooth)/ (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = loss.sum() / N
		return loss

#测试输出每一类的dice
class MulticlassDice(nn.Module):
	"""
	requires one hot encoded target. Applies DiceLoss on each class iteratively.
	requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
	  batch size and C is number of classes
	"""
	def __init__(self,num_classes):
		super(MulticlassDice, self).__init__()
		self.num_classes = num_classes

	def forward(self, input, target, weights=None):
		N = target.shape[0]
		each_mapclass=[]
		each_inputclass=[]
		eachclassdice=[]
		one_hot=get_one_hot()
		for i in range(N):
			each_map=target[i,:]
			each_map=each_map.squeeze(0)
			each_mapp=one_hot(each_map.cpu(), self.num_classes)
			each_mapp=each_mapp.unsqueeze(0)
			each_mapclass.append(each_mapp)
			each_input=input[i,:]
			each_input=each_input.squeeze(0)
			each_inputt=one_hot(each_input.cpu(), self.num_classes)
			each_inputt=each_inputt.unsqueeze(0)
			each_inputclass.append(each_inputt)
		output_target=torch.cat(each_mapclass,dim=0).cuda()
		input_class=torch.cat(each_inputclass,dim=0).cuda()
		C = input_class.shape[1]
		if weights is None:
			weights = torch.ones(C) #uniform weights for all classes
		dice = Dice()
		totalLoss = 0
		for i in range(C):
			diceLoss = dice(input_class[:,i], output_target[:,i])
			eachclassdice.append(diceLoss)
			if weights is not None:
				diceLoss *= weights[i]
			totalLoss += diceLoss
		return totalLoss,eachclassdice