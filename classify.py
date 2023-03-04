import logging
import os
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils


class Classify(object):
    def __init__(
            self,
            args,
            device,
            student,
            optimizer,
            fp16_scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule
            ) -> None:
        super().__init__()

        self.args = args
        self.device = device
        self.student = student
        self.optimizer = optimizer
        self.fp16_scaler = fp16_scaler
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.val_criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)

        # load pretrain weight
        pretrain_dict = torch.load(args.pretrain_model_path)['student']
        dict_to_load = {}
        for k, v in pretrain_dict.items():
            if k.startswith('backbone'):
                dict_to_load[k.replace('backbone.', '')] = v

        model_dict = self.student.state_dict()
        model_dict.update(dict_to_load)
        self.student.load_state_dict(model_dict)

        for n, p in self.student.named_parameters():
            if not n.startswith('head'):
                p.requires_grad = False


        self.writter = SummaryWriter(log_dir=Path(args.output_dir).joinpath("logs"))

        logging.basicConfig(filename=Path(self.writter.log_dir).joinpath('training.log'), level=logging.DEBUG)
    
    def main(self, start_epoch, train_data_loader, val_data_loader):
        for epoch in range(start_epoch, self.args.epochs):
            if '_' in self.args.ratio:
                print('请检查比例设定！')
                return

            header = 'Epoch: [{}/{}]'.format(epoch + 1, self.args.epochs)

            print(header, ' Start training ...')
            self.train(header, epoch, train_data_loader)
            print(header, ' Start validation ...')
            self.val(header, epoch, val_data_loader)

            save_dict = {
                'student': self.student.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.lr_schedule,
                'wd_schedule': self.wd_schedule,
                'momentum_schedule': self.momentum_schedule,
                'epoch': epoch + 1, 'args': self.args}
            if self.fp16_scaler is not None:
                save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()
            torch.save(save_dict, os.path.join(self.args.output_dir, 'checkpoint.pth'))
            if self.args.saveckp_freq and epoch % self.args.saveckp_freq == 0:
                torch.save(save_dict, os.path.join(self.args.output_dir, f'checkpoint{epoch:04}.pth'))
            
    
    def train(self, header, epoch, train_data_loader):
        self.student.train()
        progress_bar = utils.ProgressBar(header)
        with progress_bar.progress as progress:
            total_iters = len(train_data_loader)
            task = progress.add_task("", total=total_iters)
            progress_bar.init_time()
            for it, (images, labels) in enumerate(train_data_loader):
                progress_bar.update_iter()

                it = total_iters * epoch + it  # global training iteration
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = self.lr_schedule[it]
                    if i == 0:  
                        param_group["weight_decay"] = self.wd_schedule[it]

                # move images and labels to gpu
                images = images.to(self.device)
                labels = labels[0].to(self.device)

                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    s_cls = self.student(images, classify=True)
                    
                    top1 = utils.accuracy(s_cls, labels)[0]
                    
                    loss = self.criterion(s_cls, labels)
                    

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()), force=True)
                    sys.exit(1)

                # student update
                self.optimizer.zero_grad()
                param_norms = None
                if self.fp16_scaler is None:
                    loss.backward()
                    if self.args.clip_grad:
                        param_norms = utils.clip_gradients(self.student, self.args.clip_grad)

                    self.optimizer.step()
                else:
                    self.fp16_scaler.scale(loss).backward()
                    if self.args.clip_grad:
                        self.fp16_scaler.unscale_(self.optimizer)  
                        param_norms = utils.clip_gradients(self.student, self.args.clip_grad)

                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()

                torch.cuda.synchronize()
                meters = {
                    "loss/train_loss": loss.item(), 
                    'acc/train_acc': top1.item(), 
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'wd': self.optimizer.param_groups[0]["weight_decay"]
                    }
                
                for k, v in meters.items():
                    self.writter.add_scalar(k, v, it)
                
                progress_bar.update_task(progress, task, progress._tasks[task], meters)

                

            logging.debug('{} Train Stats: {}'.format(header, ' '.join('{}: {:.4f}'.format(item[0], item[1]) for item in meters.items())))
            progress_bar.update_total_time(progress, task, progress._tasks[task], meters)


    def val(self, header, epoch, val_data_loader):
        self.student.eval()

        val_loss = 0
        correct = 0

        progress_bar = utils.ProgressBar(header)
        with progress_bar.progress as progress:
            total_iters = len(val_data_loader)
            task = progress.add_task("", total=total_iters)
            progress_bar.init_time()
            with torch.no_grad():
                for it, (images, labels) in enumerate(val_data_loader):
                    progress_bar.update_iter()

                    it = total_iters * epoch + it  # global training iteration
                    

                    # move images and labels to gpu
                    images = images.to(self.device)
                    labels = labels[0].to(self.device)

                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        s_cls = self.student(images, classify=True)
                        
                        pred = s_cls.argmax(dim=1, keepdim=True)
                        correct += pred.eq(labels.view_as(pred)).sum().item()
                        
                        val_loss += self.val_criterion(s_cls, labels).item()  # sum up batch loss
                    meters = {
                        "acc/val_acc": 100. * correct / (it + 1),
                        "acc/val_loss": val_loss / (it + 1)
                    }
                    progress_bar.update_task(progress, task, progress._tasks[task], meters)

            torch.cuda.synchronize()
            val_loss /= (len(val_data_loader) * self.args.batch_size / 8)
            val_acc = 100. * correct / (len(val_data_loader) * self.args.batch_size / 8)
            meters = {
                "loss/val_loss": val_loss, 
                'acc/val_acc': val_acc
                }
            
            for k, v in meters.items():
                self.writter.add_scalar(k, v, total_iters * epoch)
                    
                    

                

            logging.debug('{} Val Stats: {}'.format(header, ' '.join('{}: {:.4f}'.format(item[0], item[1]) for item in meters.items())))
            progress_bar.update_total_time(progress, task, progress._tasks[task], meters)




