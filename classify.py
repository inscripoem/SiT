import os
import math
import sys
import json
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
            wd_schedule
            ) -> None:
        super().__init__()

        self.args = args
        self.device = device
        self.student = student
        self.optimizer = optimizer
        self.fp16_scaler = fp16_scaler
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.val_criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)

        # load pretrain weight
        if args.pretrain_model_path:
            pretrain_dict = torch.load(args.pretrain_model_path)['student']
            dict_to_load = {}
            for k, v in pretrain_dict.items():
                if k.startswith('backbone'):
                    if v.shape == self.student.state_dict()[k.replace('backbone.', '')].shape:
                        dict_to_load[k.replace('backbone.', '')] = v

            model_dict = self.student.state_dict()
            model_dict.update(dict_to_load)
            self.student.load_state_dict(model_dict, strict=False)

            if args.pretrain_adjust_mode == 'linear':
                print('Adjusting pretrain weight with linear mode ...')
                for n, p in self.student.named_parameters():
                    if not n.startswith('head'):
                        p.requires_grad = False
            else:
                print('Adjusting pretrain weight with finetune mode ...')

        default_tensorboard_path = Path(args.output_dir).joinpath("logs")
        self.writter = SummaryWriter(log_dir=default_tensorboard_path if not args.tensorboard_log_path else args.tensorboard_log_path)

    
    def main(self, start_epoch, train_data_loader, val_data_loader, test_data_loader):
        best_val_acc = 0.0
        for epoch in range(start_epoch, self.args.epochs):
            if '_' in self.args.ratio:
                print('请检查比例设定！')
                return

            header = 'Epoch: [{}/{}]'.format(epoch + 1, self.args.epochs)

            print(header, ' Start training ...')
            train_stats = self.train(header, epoch, train_data_loader)
            print(header, ' Start validation ...')
            val_stats = self.val(header, epoch, val_data_loader)

            save_dict = {
                'student': self.student.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.lr_schedule,
                'wd_schedule': self.wd_schedule,
                'epoch': epoch + 1, 'args': self.args}
            if self.fp16_scaler is not None:
                save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()
            torch.save(save_dict, os.path.join(self.args.output_dir, 'checkpoint.pth'))
            if self.args.saveckp_freq and epoch % self.args.saveckp_freq == 0:
                torch.save(save_dict, os.path.join(self.args.output_dir, f'checkpoint{epoch:04}.pth'))
            if val_stats['acc/val_acc'] > best_val_acc:
                best_val_acc = val_stats['acc/val_acc']
                torch.save(save_dict, os.path.join(self.args.output_dir, 'checkpoint_best.pth'))
            log_stats = {**dict(list(train_stats.items()) + list(val_stats.items())), 'epoch': epoch + 1}
            with open(os.path.join(self.args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
        print(f'Epoch {self.args.epochs} training complete. Start testing ...')
        test_stats = self.test(test_data_loader)
        print(f'Test complete. Test stats: {test_stats}')
        with open(os.path.join(self.args.output_dir, 'log_test.txt'), 'a') as f:
            f.truncate(0)
            f.write(json.dumps(test_stats))

            
    
    def train(self, header, epoch, train_data_loader):
        self.student.train()
        progress_bar = utils.ProgressBar(header)
        metric_logger = utils.MetricLogger(delimiter="  ")
        with progress_bar.progress as progress:
            total_iters = len(train_data_loader)
            task = progress.add_task("", total=total_iters)
            progress_bar.init_time()
            for it, (images, labels) in enumerate(train_data_loader):
            
                it = total_iters * epoch + it  # global training iteration
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = self.lr_schedule[it]
                    if i == 0:  
                        param_group["weight_decay"] = self.wd_schedule[it]

                # move images and labels to gpu
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    s_cls = self.student(images, classify=True)
                    
                    top1 = utils.accuracy(s_cls, labels)[0]
                    
                    loss = self.criterion(s_cls, labels)
                    

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
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
                metric_logger.update(**meters)
                progress_bar.update_iter()
                progress_bar.update_task(progress, task, progress._tasks[task], meters)

            progress_bar.update_total_time(progress, task, progress._tasks[task], meters)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def val(self, header, epoch, val_data_loader):
        self.student.eval()

        val_loss = 0
        correct = 0

        progress_bar = utils.ProgressBar(header)
        metric_logger = utils.MetricLogger(delimiter="  ")
        with progress_bar.progress as progress:
            total_iters = len(val_data_loader)
            task = progress.add_task("", total=total_iters)
            progress_bar.init_time()
            with torch.no_grad():
                for it, (images, labels) in enumerate(val_data_loader):

                    #it = total_iters * epoch + it  # global training iteration
                    

                    # move images and labels to gpu
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        s_cls = self.student(images, classify=True)
                        
                        pred = s_cls.argmax(dim=1, keepdim=True)
                        correct += pred.eq(labels.view_as(pred)).sum().item()
                        
                        val_loss += self.val_criterion(s_cls, labels).item()  # sum up batch loss
                    meters = {
                        "acc/val_loss": val_loss / ((it + 1) * val_data_loader.batch_size),
                        "acc/val_acc": 100. * correct / ((it + 1) * val_data_loader.batch_size),
                    }
                    progress_bar.update_iter()
                    progress_bar.update_task(progress, task, progress._tasks[task], meters)

            torch.cuda.synchronize()
            val_loss /= len(val_data_loader.dataset)
            val_acc = 100. * correct / len(val_data_loader.dataset)
            meters = {
                "loss/val_loss": val_loss, 
                'acc/val_acc': val_acc
                }
            
            for k, v in meters.items():
                self.writter.add_scalar(k, v, total_iters * epoch)
            metric_logger.update(**meters)
                    
            progress_bar.update_total_time(progress, task, progress._tasks[task], meters)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def test(self, test_data_loader):
        correct = 0
        total = len(test_data_loader.dataset)
        total_labels = []
        total_pred = []
        progress_bar = utils.ProgressBar('Test')
        test_model = self.student
        checkpoint = torch.load(Path(self.args.output_dir).joinpath('checkpoint.pth'))['student']
        test_model.load_state_dict(checkpoint)
        test_model.to(self.device).eval()
        with progress_bar.progress as progress:
            total_iters = len(test_data_loader)
            task = progress.add_task("", total=total_iters)
            progress_bar.init_time()
            with torch.no_grad():
                for it, (images, labels) in enumerate(test_data_loader):
                    images = images.to(self.device)
                    total_labels.append(labels)
                    labels = labels.to(self.device)
                    s_cls = test_model(images, classify=True)

                    pred = s_cls.argmax(dim=1, keepdim=True)
                    total_pred.append(pred)

                    correct += pred.eq(labels.view_as(pred)).sum().item()

                    meters = {
                            "acc/test_acc": 100. * correct / ((it + 1) * test_data_loader.batch_size),
                        }
                    progress_bar.update_iter()
                    progress_bar.update_task(progress, task, progress._tasks[task], meters)

        test_acc = 100. * correct / total

        checkpoint_best = torch.load(Path(self.args.output_dir).joinpath('checkpoint_best.pth'))['student']
        test_model.load_state_dict(checkpoint_best)
        test_model.to(self.device).eval()
        correct = 0
        total_preds = torch.tensor([], dtype=torch.long)
        total_labels = torch.tensor([], dtype=torch.long)
        progress_bar = utils.ProgressBar('Test_best')
        with progress_bar.progress as progress:
            total_iters = len(test_data_loader)
            task = progress.add_task("", total=total_iters)
            progress_bar.init_time()
            with torch.no_grad():
                for it, (images, labels) in enumerate(test_data_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    s_cls = test_model(images, classify=True)

                    pred = s_cls.argmax(dim=1, keepdim=True)

                    total_preds = torch.cat((total_preds, pred.cpu()), dim=0)
                    total_labels = torch.cat((total_labels, labels.cpu()), dim=0)

                    correct += pred.eq(labels.view_as(pred)).sum().item()

                    meters = {
                            "acc/best_test_acc": 100. * correct / ((it + 1) * test_data_loader.batch_size),
                        }
                    progress_bar.update_iter()
                    progress_bar.update_task(progress, task, progress._tasks[task], meters)
        
        best_test_acc = 100. * correct / total
        cm = torch.zeros(2, 2, dtype=torch.int)
        for i in range(len(total_labels)):
            cm[total_labels[i], total_preds[i]] += 1
        cm = cm.tolist()
        cm_string = ";".join([",".join(map(str, sublist)) for sublist in cm])
        return {'acc/test_acc': test_acc, 'acc/best_test_acc': best_test_acc, 'best_cm': cm_string}




