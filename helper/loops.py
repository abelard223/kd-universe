from __future__ import print_function, division

import sys
import time
import logging
import torch
import torch.nn.functional as F
from .util import AverageMeter, accuracy, progress_bar


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    print('\nEpoch: %d' % epoch)
    model.train()

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    train_loss = 0
    correct = 0
    total = 0 


    for batch_idx, (input, target) in enumerate(train_loader):
        # data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            
        batch_size = input.size(0)

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).sum().float().cpu()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print info
        progress_bar(batch_idx, len(train_loader), 
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


        # tensorboard logger
        pass

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        100.*correct/total))

    return 100.*correct/total, train_loss/batch_idx


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_corr = criterion_list[1]
    criterion_kd = criterion_list[2]
    if opt.distill == 'mlkd':
        criterion_align = criterion_list[3]
   
    model_s = module_list[0]
    model_t = module_list[-1]

    train_loss = 0
    correct = 0
    total = 0 

    end = time.time()
    for batch_idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        elif opt.distill in ['mlkd']:
            input, target = data
        else:
            input, target, index = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.distill not in ['mlkd']:
                index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        if opt.distill == 'mlkd':
            c,h,w = input.size()[-3:]
            input = input.view(-1, c, h, w)
            batch_size = int(input.size(0) / 4)
            nor_index = (torch.arange(4*batch_size) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch_size) % 4 != 0).cuda()

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        if opt.distill == 'mlkd':
            loss_cls = criterion_cls(logit_s[nor_index], target)
        else:
            loss_cls = criterion_cls(logit_s, target)
        loss_corr = 0
        loss_align = 0

        # other kd beyond KL divergence
        if opt.distill == 'kd' or opt.distill == 'ckd':
            loss_kd = 0
            loss_corr = criterion_corr(logit_s, logit_t)
        elif opt.distill == 'skd':
            loss_kd = 0
            loss_corr = criterion_corr(logit_s, logit_t, target) 
        elif opt.distill == 'mlkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]

            loss_corr = criterion_corr(f_s, f_t)

            aug_target = target.unsqueeze(1).expand(-1,4).contiguous().view(-1).long().cuda()
            loss_kd = criterion_kd(f_s, f_t, aug_target) 

            f_s_nor = f_s[nor_index]
            f_t_nor = f_t[nor_index]
            f_t_list = []
            for i in range(4):
                aug_index = (torch.arange(4*batch_size) % 4 == i).cuda()
                f_t_aug = f_t[aug_index]
                f_t_list.append(f_t_aug)
            loss_align = criterion_align(f_s, f_t)
    
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_corr + opt.beta * loss_kd + opt.delta * loss_align
        train_loss += loss.item()
        
        if opt.distill == 'mlkd':
            _, predicted = torch.max(logit_s[nor_index], 1)
        else:
            _, predicted = torch.max(logit_s, 1)
            
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum().float()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        progress_bar(batch_idx, len(train_loader), 
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        100.*correct/total))
    
    return 100.*correct/total, train_loss/batch_idx


def validate(epoch, val_loader, model, criterion, opt):
    """validation"""
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):

            #if opt.distill == 'mlkd':
             #   input = input[:,0,:,:,:]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum().float()

            progress_bar(batch_idx, len(val_loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc))

    return acc, val_loss/(batch_idx+1)
