import argparse
import torch
import json
import os
import numpy as np
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms ,models

def arg_parser():
    parser = argparse.ArgumentParser(description="Predict Settings")
    parser.add_argument('dir',type=str,help='distation of image')  
    parser.add_argument('checkpoint',default=os.getcwd()+'/checkpointTrPy.pth',type=str,help='checkpoint path')
    parser.add_argument('--top_k',default=3,type=int,help='retrun k prob')
    parser.add_argument('--cat_path',default=os.getcwd()+'/cat_to_name.json',type=str,help='Number of Node in hideing layer')
    parser.add_argument('--gpu',action="store_true",help='Use GPU + Cuda for fast trainning')
   
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    print('load_checkpoint')
    checkpoint = torch.load(filepath)
    exec("m = models.{}(pretrained=True)".format(checkpoint['arch']))
    model=(locals()["m"])
    for param in model.parameters(): 
        param.requires_grad = False

    model.class_to_idx = checkpoint['cls_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    print('end load_checkpoint')
    return model

def load_cat_name(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    print('process_image')
    im=Image.open(image)
    h,w=im.height,im.width
    
    if h >256 and w >256:
        if h>w:
            h=int(h*256/w)
            w=256
        else:
            w=int(w*256/h)
            h=256
    
    im.thumbnail((w,h))
    np_im=np.array(im)/255
    x=int((np_im.shape[0]-224)/2)
    y=int((np_im.shape[1]-224)/2)
    np_im = np_im[x:x+224 , y:y+224]
    n_mean = np.array([0.485, 0.456, 0.406])
    n_std = np.array([0.229, 0.224, 0.225])
    np_im = (np_im-n_mean)/n_std
    np_im = np_im.transpose(2, 0, 1)
    print('end process_image')
    return np_im

def predict(image_path, model, topk,device):
    print('predict')
    if device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device='cpu'
        
    print('you work with {} ....'.format(device))

    image = torch.from_numpy(np.expand_dims(process_image(image_path),axis=0)).type(torch.FloatTensor).to(device)
    
    probs = torch.exp(model.forward(image))
    top_k_probs, classes = probs.topk(topk)
    
    top_k_probs = np.array(top_k_probs.detach())[0]
    classes  = np.array(classes.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes   = [idx_to_class[lab] for lab in classes]
    print('end predict')
    return top_k_probs, classes

def main():
    args = arg_parser()
    print('main')

    model = load_checkpoint(args.checkpoint)
    top_k_probs, classes = predict(args.dir ,model,args.top_k,args.gpu)
    cat_to_name=load_cat_name(args.cat_path)
    for k,c in zip(top_k_probs, classes):
        print('the class name is :{} and probabltiy is :{:.2f}'.format(cat_to_name[str(c)],k))
    
    print("\done...!!")
    
if __name__ == '__main__': main()
