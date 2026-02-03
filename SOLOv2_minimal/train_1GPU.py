#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import datetime
from torch.nn.utils import clip_grad

from model.solov2 import SOLOv2
from configs import *
from utils import timer
from data_loader.build_loader import make_data_loader
from val import val


def main():
    device = torch.device("cuda:0")

    # --------------------------
    # Config
    # --------------------------
    cfg = Custom_nori_res50(mode='train')
    cfg.print_cfg()

    # --------------------------
    # Model
    # --------------------------
    model = SOLOv2(cfg).to(device)
    model.train()

    # --------------------------
    # DataLoader
    # --------------------------
    data_loader = make_data_loader(cfg)
    len_loader = len(data_loader)
    max_iter = len_loader * cfg.epochs
    print(f'Length of dataloader: {len_loader}, total iterations: {max_iter}')

    # --------------------------
    # Resume / LR
    # --------------------------
    start_epoch = 1
    step = 1
    start_lr = cfg.lr

    if cfg.break_weight:
        start_epoch = int(cfg.break_weight.split('_')[-1][:-4]) + 1
        step = (start_epoch - 1) * len_loader + 1
        model.load_state_dict(torch.load(cfg.break_weight, map_location=device))
        print(f'Continue training from {cfg.break_weight}')

        for e in cfg.lr_decay_steps:
            if start_epoch > e:
                start_lr *= 0.1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=start_lr,
        weight_decay=0.0005
    )

    timer.reset(reset_at=step)

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(start_epoch, cfg.epochs + 1):
        for img, gt_labels, gt_bboxes, gt_masks in data_loader:
            timer.start(step)

            img = img.to(device, non_blocking=True)
            gt_labels = [g.to(device) for g in gt_labels]
            gt_bboxes = [g.to(device) for g in gt_bboxes]

            # --------------------------
            # Warmup LR
            # --------------------------
            if cfg.warm_up_iters > 0 and step <= cfg.warm_up_iters:
                lr = (cfg.lr - cfg.warm_up_init) * step / cfg.warm_up_iters + cfg.warm_up_init
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # --------------------------
            # LR decay
            # --------------------------
            if epoch in cfg.lr_decay_steps:
                decay = cfg.lr_decay_steps.index(epoch) + 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * (0.1 ** decay)

            # --------------------------
            # Forward + Loss
            # --------------------------
            with timer.counter('for+loss'):
                loss_cate, loss_ins = model(img, gt_labels, gt_bboxes, gt_masks)
                loss_total = loss_cate + loss_ins

            # --------------------------
            # Backward
            # --------------------------
            with timer.counter('backward'):
                optimizer.zero_grad()
                loss_total.backward()
                clip_grad.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=35
                )

            # --------------------------
            # Update
            # --------------------------
            with timer.counter('update'):
                optimizer.step()

            timer.add_batch_time()

            # --------------------------
            # Log
            # --------------------------
            if step % 50 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                t_t, t_d, t_fl, t_b, t_u = timer.get_times(
                    ['batch', 'data', 'for+loss', 'backward', 'update']
                )
                seconds = (max_iter - step) * t_t
                eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                print(
                    f'epoch:{epoch} step:{step} | '
                    f'lr:{cur_lr:.2e} | '
                    f'l_cls:{loss_cate.item():.3f} '
                    f'l_ins:{loss_ins.item():.3f} | '
                    f't:{t_t:.3f}s | ETA:{eta}'
                )

            step += 1

        # --------------------------
        # Save + Val
        # --------------------------
        if epoch % cfg.val_interval == 0:
            if epoch > cfg.start_save:
                save_path = f'weights/{cfg.name()}_{epoch}.pth'
                torch.save(model.state_dict(), save_path)
                print(f'Saved: {save_path}')

            val(cfg, model)
            cfg.train()
            model.train()
            timer.reset(reset_at=step)


if __name__ == '__main__':
    main()
