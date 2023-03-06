import os
import math
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision

import utils


class SiT(object):
    def __init__(
            self,
            args,
            device,
            student,
            teacher,
            optimizer,
            fp16_scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule
            ) -> None:
        super().__init__()

        # prepare SimCLR methods
        self.simclr_methods = SimCLR(args.simclr_temp).to(device)

        self.args = args
        self.device = device
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.fp16_scaler = fp16_scaler
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule

        default_tensorboard_path = Path(args.output_dir).joinpath("logs")
        self.writter = SummaryWriter(log_dir=default_tensorboard_path if not args.tensorboard_log_path else args.tensorboard_log_path)

    
    def train(self, start_epoch, trainval_data_loader):
        for epoch in range(start_epoch, self.args.epochs):
            save_recon = os.path.join(self.args.output_dir, 'reconstruction_samples')
            Path(save_recon).mkdir(parents=True, exist_ok=True)
            bz = self.args.batch_size
            plot_ = True if epoch % 10 == 0 else False

            header = 'Epoch: [{}/{}]'.format(epoch + 1, self.args.epochs)
            progress_bar = utils.ProgressBar(header)
            metric_logger = utils.MetricLogger(delimiter="  ")
            with progress_bar.progress as progress:
                total_iters = len(trainval_data_loader)
                task = progress.add_task("", total=total_iters)
                progress_bar.init_time()
                for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(trainval_data_loader):
                    

                    it = total_iters * epoch + it  # global training iteration
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group["lr"] = self.lr_schedule[it]
                        if i == 0:  
                            param_group["weight_decay"] = self.wd_schedule[it]

                    # move images to gpu
                    clean_crops = [im.to(self.device, non_blocking=True) for im in clean_crops]
                    corrupted_crops = [im.to(self.device, non_blocking=True) for im in corrupted_crops]
                    masks_crops = [im.to(self.device, non_blocking=True) for im in masks_crops]

                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        t_cls, _ = self.teacher(torch.cat(clean_crops[0:]), recons=False) 
                        s_cls, s_recons = self.student(torch.cat(corrupted_crops[0:]))
                        
                        c_loss = self.simclr_methods.total_c_loss(s_cls, t_cls, epoch)
                        top1, top5 = self.simclr_methods.top_k_acc(s_cls, (1,5))
                        
                        #-------------------------------------------------
                        recloss = F.l1_loss(s_recons, torch.cat(clean_crops[0:]), reduction='none')
                        r_loss = recloss[torch.cat(masks_crops[0:2])==1].mean() 
                            
                        if plot_==True:
                            plot_ = False
                            #validating: check the reconstructed images
                            print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg' 
                            imagesToPrint = torch.cat([clean_crops[0][0: min(15, bz)].cpu(),  corrupted_crops[0][0: min(15, bz)].cpu(),
                                                s_recons[0: min(15, bz)].cpu(), masks_crops[0][0: min(15, bz)].cpu()], dim=0)
                            # save images to tensorboard
                            self.writter.add_image('reconstruction_samples', torchvision.utils.make_grid(imagesToPrint, nrow=min(15, bz), normalize=True, range=(-1, 1)), epoch)
                            torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, bz), normalize=True, range=(-1, 1))
                        
                        
                        loss = c_loss + self.args.lmbda * r_loss

                        # Calculate psnr and ssim
                        psnr, ssim = utils.calculate_psnr_ssim(s_recons[0:].cpu(), torch.cat(clean_crops[0:]).cpu())

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

                    # EMA update for the teacher
                    with torch.no_grad():
                        m = self.momentum_schedule[it]  # momentum parameter
                        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                    # logging
                    torch.cuda.synchronize()
                    meters = {"loss/total": loss.item(), 
                              "loss/c_loss": c_loss.item(), 
                              "loss/r_loss": r_loss.item(), 
                              'acc/top1': top1.item(), 
                              'acc/top5': top5.item(), 
                              'psnr': psnr, 
                              'ssim':ssim,
                              'lr': self.optimizer.param_groups[0]["lr"],
                              'wd': self.optimizer.param_groups[0]["weight_decay"]}
                    
                    for k, v in meters.items():
                        self.writter.add_scalar(k, v, it)
                    
                    metric_logger.update(**meters)

                    progress_bar.update_iter()
                    progress_bar.update_task(progress, task, progress._tasks[task], meters)

                progress_bar.update_total_time(progress, task, progress._tasks[task], meters)

            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            save_dict = {
                'student': self.student.state_dict(),
                'teacher': self.teacher.state_dict(),
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
            
            log_stats = {**train_stats, 'epoch': epoch + 1}
            with open(os.path.join(self.args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

            




class SimCLR(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        
        self.temp = temp
        
    def contrastive_loss(self, q, k):
        
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        N = logits.shape[0] 
        
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)

    def total_c_loss(self, student_output, teacher_output, epoch):

        student_out = student_output
        student_out = student_out.chunk(2)

        teacher_out = teacher_output 
        teacher_out = teacher_out.detach().chunk(2)

        return self.contrastive_loss(student_out[0], teacher_out[1]) + self.contrastive_loss(student_out[1], teacher_out[0])

    def top_k_acc(self, features_out, topk=(1,)):
        features = features_out.detach()

        labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temp
        
        return utils.accuracy(logits, labels, topk)