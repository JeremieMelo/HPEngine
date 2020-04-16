#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-12 19:52:41
@LastEditTime: 2019-08-23 14:09:43
'''
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

from args_config import args
from model import *
from utils import *

if (torch.cuda.is_available() and args.use_cuda):
    torch.cuda.set_device(args.gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    device = torch.device('cuda:'+str(args.gpu_id))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    torch.backends.cudnn.benchmark = False

torch.autograd.set_detect_anomaly(True)

if(args.deterministic == True):
    set_torch_deterministic()

batch_size = args.batch_size
if(args.dataset == "mnist"):
    train_dataset = datasets.MNIST('./data',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(
                                        (args.img_height, args.img_width), interpolation=2), transforms.ToTensor()
                                ])
                                )

    validation_dataset = datasets.MNIST('./data',
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(
                                                (args.img_height, args.img_width), interpolation=2), transforms.ToTensor()
                                        ])
                                        )
if(args.dataset == "fashionmnist"):
    train_dataset = datasets.FashionMNIST('./data',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(
                                        (args.img_height, args.img_width), interpolation=2), transforms.ToTensor()
                                ])
                                )

    validation_dataset = datasets.FashionMNIST('./data',
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(
                                                (args.img_height, args.img_width), interpolation=2), transforms.ToTensor()
                                        ])
                                        )
elif(args.dataset == "cifar10"):
    train_dataset = datasets.CIFAR10('./data',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(
                                        (args.img_height, args.img_width), interpolation=2), transforms.ToTensor()
                                ])
                                )

    validation_dataset = datasets.CIFAR10('./data',
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(
                                                (args.img_height, args.img_width), interpolation=2), transforms.ToTensor()
                                        ])
                                        )


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
if(args.model == "HP_CLASS_CNN"):
    model = HP_CLASS_CNN(img_height=args.img_height,
                img_width=args.img_width,
                in_channels=args.in_channels,
                n_class=args.n_class,
                hidden_list=args.hidden_list,
                kernel_list=args.kernel_list,
                pool_out_size=args.pool_out_size,
                in_bits=args.input_bit,
                w_bits=args.weight_bit,
                act=args.act,
                act_thres=args.act_thres,
                mode=args.mode,
                device=device).to(device)
elif(args.model == "HP_CLASS_CNN2"):
    model = HP_CLASS_CNN2(img_height=args.img_height,
                img_width=args.img_width,
                in_channels=args.in_channels,
                n_class=args.n_class,
                hidden_list=args.hidden_list,
                kernel_list=args.kernel_list,
                pool_out_size=args.pool_out_size,
                in_bits=args.input_bit,
                w_bits=args.weight_bit,
                act=args.act,
                act_thres=args.act_thres,
                mode=args.mode,
                input_augment=False,
                device=device).to(device)
elif(args.model == "HP_CLASS_CNN3"):
    model = HP_CLASS_CNN3(img_height=args.img_height,
                img_width=args.img_width,
                in_channels=args.in_channels,
                n_class=args.n_class,
                hidden_list=args.hidden_list,
                kernel_list=args.kernel_list,
                pool_out_size=args.pool_out_size,
                in_bits=args.input_bit,
                w_bits=args.weight_bit,
                act=args.act,
                act_thres=args.act_thres,
                mode=args.mode,
                device=device).to(device)




optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=args.lr_gamma)
# thres_scheduler = ThresholdScheduler(args.phase_2, args.epoch-1, 0.05, 1, mode="tanh")
phase_noise_scheduler = ThresholdScheduler_tf(0, args.epoch, args.phase_noise_std, args.phase_noise_std/3)
disk_noise_scheduler = ThresholdScheduler_tf(0, args.epoch, args.disk_noise_std, args.disk_noise_std/3)
value_reg = ValueRegister(operator=lambda x, y: x if x >
                          y else y, name="Best Accuracy", show=True)
value_tracer = ValueTracer(show=False)
saver = BestKModelSaver(k=args.save_best_model_k)
writer = SummaryWriter()
lg = Logger().logger
criterion = F.nll_loss

# print(model)
print(f'[I] Number of parameters: {count_parameters(model)}')
# summary_model(model, [(batch_size, args.in_channels, args.img_height, args.img_width)])

model_name = f"{args.model}_{args.img_height}x{args.img_width}-{'-'.join(['c'+str(i) for i in args.kernel_list])}-{'-'.join(['f'+str(i) for i in args.hidden_list])}-f{args.n_class}_mode-{args.mode}_wb-{args.weight_bit}_ib-{args.input_bit}"
checkpoint = f"./checkpoint/{args.checkpoint_dir}/{model_name}_{args.model_comment}.pt"
lg.info(checkpoint)
epochs = args.epoch


def train(phase, epoch, prune_phases=False, log_interval=args.log_interval, train_mode="unaware", phase_noise_std=0, disk_noise_std=0):

    model.train()
    step = epoch * len(train_loader)
    correct = 0
    lasso_loss = None
    unitary_loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if(train_mode in {"noise", "calibrate"}):
            model.assign_engines(args.out_par, args.batch_par, phase_noise_std, disk_noise_std, deterministic=False)
        if(train_mode == "calibrate"):
            model.static_pre_calibration()
        data = data.to(device)

        target = target.to(device)
        optimizer.zero_grad()
        model.zero_grad()


        output = model(data)


        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().cpu()
        # Calculate loss
        class_loss = criterion(output, target)
        # class_loss = torch.zeros(1).to(device)
        loss = class_loss

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        step += 1
        # break


        if batch_idx % log_interval == 0:
            lg.info('Train Phase: {} Epoch: {} [{:6d}/{:6d} ({:3.0f}%)] Class: {:.4f}'.format(phase, epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), class_loss.data.item()))



    writer.add_scalar("data/train_loss", loss.data.item(), step)
    accuracy = 100. * correct.to(torch.float32) / len(train_loader.dataset)
    lg.info('Train Phase: {} Epoch: {} Accuracy: {:.2f}%'.format(
        phase, epoch, accuracy.data.item()))


def validate(epoch, loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    # print(model.layers["conv1"].weights)

    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        val_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        # break


    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / \
        len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
         val_loss, correct, len(validation_loader.dataset), accuracy))

    value_reg.register_value(accuracy)
    writer.add_scalar("data/val_loss", val_loss, epoch)
    writer.add_scalar("data/val_acc", accuracy, epoch)



if __name__ == "__main__":

    lossv, accv = [], []
    epoch = 0
    try:
        phase = 1
        epoch = 0
        ### test full precision model ###
        lg.info("Test full-precision model")
        load_model(model, checkpoint)

        model.disable_calibration()

        for i in range(20):
            model.assign_engines(args.out_par, args.batch_par, args.phase_noise_std, args.disk_noise_std, deterministic=False)
            if(args.robust_assign):
                # model.robust_reassign_engines()
                # model.robust_reassign_engines_fine()
                model.static_pre_calibration()
                model.enable_calibration()
            if(args.mode == "ringonn"):
                model.assign_ring_noise(args.disk_noise_std, 0.005)
            validate(epoch, lossv, accv)
        lg.info(f"Average loss: {np.mean(lossv)}, Average Acc: {np.mean(accv):.2f}%, Std Acc: {np.std(accv):.3f}")
        accv.append(np.mean(accv))

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")
    writer.close()


