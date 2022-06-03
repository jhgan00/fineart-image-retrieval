# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from metric import mean_average_precision, precision_at_k
from torch.nn import functional as F

import mae.util.misc as misc
import mae.util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, styles, genres) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        styles = styles.to(device, non_blocking=True)
        genres = genres.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():

            features = model.forward_features(samples)
            outputs1, outputs2 = model.head1(features), model.head2(features)
            style_loss = criterion(outputs1, styles)
            genre_loss = criterion(outputs2, genres)

            features = F.normalize(features, dim=1)

            # distance 계산
            dist_matrix = torch.cdist(features.unsqueeze(0), features.unsqueeze(0)).squeeze(0)

            # knn 찾기
            knn_dist, idx = dist_matrix.topk(k=args.k+1, dim=-1, largest=False)
            knn_dist, idx = knn_dist[:, 1:], idx[:, 1:]

            # style, genre hit 여부 계산
            style_correct = styles[idx] == styles.unsqueeze(dim=-1)
            genre_correct = genres[idx] == genres.unsqueeze(dim=-1)

            # 실제 라벨 - pos:1 gray:0.5 neg:0
            correct = (style_correct.type(torch.int) + genre_correct.type(torch.int)) * 0.5

            # 로스 계산 ver2
            # 순서가 올바른 경우에도 margin 을 만족하지 않는 경우는 로스 적용
            # 하나의 앵커에서 전부 평균. 로스 0인 경우도 포함.
            knn_loss = torch.tensor(0.0, requires_grad=True).to(args.device)

            for i in range(args.k):
                d_close = knn_dist[:, i:i + 1] # 가까운 데이터 포인트 (positive 역할)
                d_far   = knn_dist[:, i + 1:]  # 먼 데이터 포인트 (negative 역할)

                # 가까운 데이터 포인트의 라벨, 먼 데이터 포인트의 라벨 구하고 둘의 차이 구하기
                # example. anchor - negative(0.0) - neutral(0.5) 과 같은 순서로 배열된 경우
                # label_diff = 0.5 - 0.0 = 0.5 -> 0보다 크면 잘못된 배열
                # example. anchor - positive(1.0) - negative(0.0) 과 같은 순서로 배열된 경우
                # label_diff = 0.0 - 0.1 = -1.0 -> 0보다 같거나 작으면 올바른 배열
                close_label = correct[:, i:i + 1]
                far_label = correct[:, i + 1:]
                label_diff = far_label - close_label

                # 1. invalid_rank_loss: 먼 데이터의 라벨보다 가까운 데이터의 라벨이 작다면 incorrect
                invalid_mask = (label_diff > 0).type(torch.int)
                # d_pos - d_neg + margin : margin 은 일치도 여부에 따라 다르게 적용
                invalid_rank_loss = torch.clamp(d_far - d_close + args.margin * label_diff, min=0) * invalid_mask

                # 2. invalid_margin_loss: 순서는 맞지만 margin 을 만족하지 못하는 경우
                valid_mask = (label_diff <= 0).type(torch.int)
                invalid_margin_loss = torch.clamp(d_close - d_far + args.margin * -label_diff, min=0) * valid_mask

                # 3. 두 로스를 합하면 최종 로스
                loss_at_k = (invalid_rank_loss + invalid_margin_loss).sum()
                knn_loss = knn_loss + loss_at_k

            knn_loss = knn_loss / (args.k * (args.k - 1) * 0.5) / args.batch_size
            loss = args.lambda_style * style_loss + args.lambda_genre * genre_loss + args.lambda_knn * knn_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = loss_value #misc.all_reduce_mean(loss_value, device=device)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/total', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss/style', style_loss * args.lambda_style, epoch_1000x)
            log_writer.add_scalar('loss/genre', genre_loss * args.lambda_genre, epoch_1000x)
            log_writer.add_scalar('loss/knn', knn_loss * args.lambda_knn, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(device)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    db = []
    style_label = []
    genre_label = []

    for (images, styles, genres) in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        styles = styles.to(device, non_blocking=True)
        genres = genres.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            features = model.forward_features(images)
            features = F.normalize(features, dim=1)

            db.append(features)
            style_label += styles
            genre_label += genres

    db = torch.cat(db, dim=0)
    style_label = torch.tensor(style_label)
    genre_label = torch.tensor(genre_label)

    mAP_style = mean_average_precision(db, style_label, rank=10)
    mAP_genre = mean_average_precision(db, genre_label, rank=10)

    p_at_k_style = precision_at_k(db, style_label, rank_list=[1, 5, 10])
    p_at_k_genre = precision_at_k(db, genre_label, rank_list=[1, 5, 10])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(device)
    print("* STYLE mAP {0:.2%} P@1 {1:.2%} P@5 {2:.2%} P@10 {3:.2%}".format(mAP_style, *p_at_k_style))
    print("* GENRE mAP {0:.2%} P@1 {1:.2%} P@5 {2:.2%} P@10 {3:.2%}".format(mAP_genre, *p_at_k_genre))

    if log_writer is not None:
        log_writer.add_scalar("metric/style/mAP", mAP_style, epoch)
        log_writer.add_scalar("metric/style/P@1", p_at_k_style[0], epoch)
        log_writer.add_scalar("metric/style/P@5", p_at_k_style[1], epoch)
        log_writer.add_scalar("metric/style/P@10", p_at_k_style[2], epoch)

        log_writer.add_scalar("metric/genre/mAP", mAP_genre, epoch)
        log_writer.add_scalar("metric/genre/P@1", p_at_k_genre[0], epoch)
        log_writer.add_scalar("metric/genre/P@5", p_at_k_genre[1], epoch)
        log_writer.add_scalar("metric/genre/P@10", p_at_k_genre[2], epoch)

    eval_stats = [mAP_style, mAP_genre] + p_at_k_style + p_at_k_genre

    return eval_stats
