import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np
import cv2
from torchvision import transforms as T
from PIL import Image
import shutil


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def plot(data, output_path, y_label, title, color='r', x_label="epoch"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, color, label=y_label)
    ax.set_title(title )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()  
    plt.legend()
    plt.savefig(os.path.join(output_path, "{}.png".format(title)))
    plt.close()

def plot_w(ops, output_path,title,y_label, x_label="epoch"):
    	
	el1,el2,no,ve,fusion=[],[],[],[],[]
	for i,op in enumerate(ops):
		el1.append(ops[i][0])
		el2.append(ops[i][1])
		no.append(ops[i][2])
		ve.append(ops[i][3])
	total=[el1,el2,no,ve]
	name=['A','B','C','D']
  		
	colors=['black','red','green','blue','yellow','m','c','pink']
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i,op in enumerate(total):
		ax.plot(op, colors[i], label=name[i])

	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	plt.grid()  
	plt.legend()
	plt.savefig(os.path.join(output_path, "{}.png".format(title)))
	plt.close()

def plot_w_acc(ops,val_acc, output_path,title,y_label, x_label="epoch"):
    	
    el1,el2,no,ve,fusion=[],[],[],[],[]
    for i,op in enumerate(ops):
        el1.append(ops[i][0])
        el2.append(ops[i][1])
        no.append(ops[i][2])
        ve.append(ops[i][3])
    total=[el1,el2,no,ve]
    name=['A','B','C','D']
        
    colors=['red','black','green','blue','yellow','m','c','pink']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    acc=list(np.array(val_acc)/100)
    ax.plot(acc, 'red', label='val_acc')
    for i,op in enumerate(total):
        ax.plot(op, colors[i+1], label=name[i])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()  
    plt.legend()
    plt.savefig(os.path.join(output_path, "{}.png".format(title)))
    plt.close()

def plot_acc(data1,data2, output_path, y_label1,y_label2,title, color1='r',color2='b', x_label="epoch"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data1, color1, label=y_label1)
    ax.plot(data2, color2, label=y_label2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(title)
    plt.grid()  
    plt.legend()
    plt.savefig(os.path.join(output_path, "{}.png".format(title)))
    plt.close()



def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)


def normalize(im,mode=1):
	"""
	Normalize volume's intensity to range [0, 1], for suing image processing
	Compute global maximum and minimum cause cerebrum is relatively homogeneous
	"""
	if mode==0:
		_max = np.max(im)
		_min = np.min(im)
		im=(im - _min) / (_max - _min)
	elif mode==1:
		im=im/255
	else:
		raise ValueError("mode is only 0 or 1")
	# `im` with dtype float64
	return im


def remove(file_path):
	if os.path.exists(file_path):  # 如果文件存在
		os.remove(file_path)
	else:
		pass

def standandzation(image):
	mean=np.mean(image)
	std=np.std(image)
	image=(image-mean)/(std+1e-4)
	return image


def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = -1, keepdims = True)
    s = x_exp / x_sum    
    return s
