import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms ,models
from collections import OrderedDict 
from workspaceutils import active_session
import os

def arg_parser():
    parser = argparse.ArgumentParser(description="NN Settings")
    parser.add_argument('dir',type=str,help='distation of image folder')  
    parser.add_argument('--arch',default='vgg16',type=str,help='choose your architecture model')
    parser.add_argument('--p_every',default=50,type=int,help='result evry ? cycle')
    parser.add_argument('--hide_layer',default=1000,type=int,help='Number of Node in hideing layer')
    parser.add_argument('--output',default=102,type=int,help='Number of output class')
    parser.add_argument('--gpu',action="store_true",help='Use GPU + Cuda for fast trainning')
    
    parser.add_argument('--save_dir',default=os.getcwd(),type=str,help='save ckeck point at ?')   
    parser.add_argument('--l_rate',default=.001,type=float,help='learnnig rate')      
    parser.add_argument('--epochs',default=3,type=int,help='cycle of tranning')  
    
    args = parser.parse_args()
    return args
    
def load_data(i_dir ,tranning =True):
    if tranning :
        trans = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        i_dataset = datasets.ImageFolder(i_dir, transform=trans)
        i_loaders = torch.utils.data.DataLoader(i_dataset,batch_size=50,shuffle=True)
    else:
        trans = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        i_dataset = datasets.ImageFolder(i_dir, transform=trans)
        i_loaders = torch.utils.data.DataLoader(i_dataset,batch_size=50)
    return i_loaders,i_dataset

def train_network(arch,h_layer,output):
    exec("m = models.{}(pretrained=True)".format(arch))
    model=(locals()["m"])
    model.name=arch
    for p in model.parameters():
        p.requires_grad=False 

    
    if str(type(model.classifier))=="<class 'torch.nn.modules.container.Sequential'>":
        print('Sequential')
        n_input = model.classifier[0].in_features
    elif str(type(model.classifier))=="<class 'torch.nn.modules.linear.Linear'>":
        print('Linear')
        n_input = model.classifier.in_features
    
    c=nn.Sequential(OrderedDict([('fc1',nn.Linear(n_input, h_layer)),('relu1',nn.ReLU()),('dropout1',nn.Dropout(p=0.5)),
                                 ('fc2',nn.Linear(h_layer, output)),('output',nn.LogSoftmax(dim=1))
                            ]))   
    model.classifier=c 
    return model

def validat(model,loader,criterion,device):
    test_loss = 0
    accuracy = 0
    for images, labels in loader: 
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy    

def training(model, train_dir, valid_dir, device, epochs, print_every,l_r,saveto):
    train_loaders ,train_dataset=load_data(train_dir)
    valid_loaders,valid_dataset=load_data(valid_dir,tranning=False)
    
    criterion = nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=l_r)
    
    steps  =0
    r_loss =0
    if device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device='cpu'
    print('you work with {} ....'.format(device))
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        step=0
        for imgs, lbls in train_loaders:
            steps += 1
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model.forward(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            with active_session():
                if steps % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, accuracy = validat(model, valid_loaders, criterion,device)
                    step+=1
                    print("Step/Epoch/Total: {}/{}/{}...".format(step,e+1, epochs),
                          "Training loss: {:.3f}...".format(running_loss/print_every),
                          "Validation loss: {:.3f}...".format(valid_loss/len(valid_loaders)),
                          "Accuracy: {:.3f}".format(accuracy/len(valid_loaders)))
                    running_loss = 0
                    model.train()
    
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'arch': model.name,
                  'classifier': model.classifier,
                  'cls_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, saveto+'/checkpointTrPy.pth')
    print('\nTraining is done and save as checkpointTrPy.pth ...')

def main():
    args = arg_parser()

    data_dir=args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    model = train_network(arch=args.arch,h_layer=args.hide_layer,output=args.output)
    
    trained_model = training(model, train_dir, valid_dir,args.gpu,  args.epochs,args.p_every,args.l_rate,args.save_dir)
    
    print("\done...!!")
    
if __name__ == '__main__': main()    
    