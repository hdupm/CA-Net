import torch
import socket
import time
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from CANet import mynet
#from NEW_ARCH import FIVE_APLUSNet
from torch.autograd import Variable
from data import get_training_set
import os 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch._dynamo
import tqdm as tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CA-Net')
parser.add_argument('--device', type=str, default='cuda:0')# help='device assignment ("cpu" or "cuda")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1680, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=4e-4, help='Learning Rate. Default=4e-4')
parser.add_argument('--data_dir', type=str, default='/mnt/e/Datas/UIEB6.4/train')#数据集位置
parser.add_argument('--label_train_dataset', type=str, default='GT/')#GT位置
parser.add_argument('--data_train_dataset', type=str, default='IN/')#input位置
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--best_save_folder', default='best_weights/', help='Location to save best checkpoint models')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='10000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_augmentation', type=bool, default=True)

writer = SummaryWriter("./logs_train")

opt = parser.parse_args()
device = torch.device(opt.device)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    model.cuda()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.to(device)
            target = target.to(device)
        #print(f"Input device is : cuda:{input.get_device()}")
        #print(f"Targrt device is : cuda:{target.get_device()}")
        t0 = time.time()        
        model.forward(input).cuda(0)
        loss = model.elbo(target)
        #creation=CharbonnierLoss()
        #loss=creation(target)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        t1 = time.time()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch, iteration, 
                          len(training_data_loader), loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    writer.add_scalar("train_loss", epoch_loss / len(training_data_loader), epoch)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
    return epoch_loss / len(training_data_loader)


def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def best_checkpoint(epoch):
    model_out_path = opt.best_save_folder+"best.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_set = get_training_set(opt.data_dir, opt.label_train_dataset, opt.data_train_dataset, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
#########################################################################
# load data of train
# for data in tqdm(training_data_loader,position=0,desc='load train dataset'):
#     pass
# training_data_loader.dataset.set_use_cache(use_cache=True)
# training_data_loader.num_workers = 2
#########################################################################
torch.cuda.set_device(0)
#torch._dynamo.config.suppress_errors = True

model = mynet(opt).eval().cuda()
#model.load_state_dict(torch.load('weights/epoch_1680.pth'))
#model = torch.compile(model)

# print('---------- Networks architecture -------------')
# print_network(model)
# print('----------------------------------------------')

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)


#optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
#base_lr = 1e-4
#max_lr =  5e-4

#设置 学习率调节方法
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=500, step_size_down=500, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)


optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-8)
#scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
#scheduler=lrs.CyclicLR(optimizer,base_lr=0.1,max_lr=0.2,step_size_up=30,step_size_down=10,cycle_momentum=False)
base_lr = 4e-4
max_lr =  1e-4
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=30, step_size_down=30, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.999, last_epoch=-1)

lowest_loss = 100

for epoch in range(opt.start_iter, opt.nEpochs + 1):

    loss = train(epoch)
    # if epoch > 300 and loss < lowest_loss:
    #     lowest_loss = loss
    #     best_checkpoint(epoch)
    #     print('Lowest loss: {:.4f}'.format(lowest_loss))

    scheduler.step()

    if (epoch+1) % opt.snapshots == 0:
        checkpoint(epoch)

writer.close()

