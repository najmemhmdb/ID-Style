import os
import copy
import time
import datetime
from munch import Munch
import numpy as np
import torchvision.utils as vutils
from metrics.id_loss import IDLoss
from torchvision import transforms
import torchvision
import math
import gc
from adabelief_pytorch import AdaBelief
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from core.model import build_model, IDStyle
from core.data_loader import (
    InputFetcher,
    get_data_loader
)
from models import StyleGAN2Generator
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# comment following packages while runing test.sh 
import core.utils as utils
from core.utils import write_in_tensorboard, labels_target_stage1, toggle_lab
from torch.utils.tensorboard import SummaryWriter


def binary_loss(logit, target, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="mean", pos_weight=pos_weight)
    return loss


def compute_g_loss(itr,
        model,
        attr_classifier,
        gan_model,
        args,
        img_src,
        lat,
        lab_src,
        lab_trg,
        frs = None,
        iter=None
):
    attr_idx = iter % 4
    outputs = {}
    all_losses = Munch()
    g_loss = 0
    outputs['all'], mul = model(lat, lab_trg, iter)

    imgs_trg = []
    for attr in range(4):
        imgs_trg.append(gan_model(outputs['all'][attr]).unsqueeze(0))
    
    imgs_trg = torch.cat(imgs_trg, dim=0)

    # classification loss 
    cls_losses = []
    att_we = [1, 1, 1, 1]
    for idx in range(4):
        fake_cls, _ = attr_classifier(imgs_trg[idx])
        target_label = toggle_lab(lab_src, idx)[:, idx]
        target_label = torch.abs(target_label - 0.1)
        loss_cls = binary_loss(fake_cls[:, idx], toggle_lab(lab_src, idx)[:, idx]) * att_we[attr_idx]
        cls_losses.append(loss_cls.unsqueeze(0))

    loss_cls_total = torch.mean(torch.cat(cls_losses),dim=0) * args.lambda_cls
    all_losses.cls = loss_cls_total.item()
    all_losses.gender = cls_losses[0].item()
    all_losses.glasses = cls_losses[1].item()
    all_losses.age = cls_losses[2].item()
    all_losses.smile = cls_losses[3].item()
    g_loss += loss_cls_total

    # sparsity loss
    if args.lambda_sparsity > 0:
        params = []
        params.append(model.iaip.gender_shared)
        params.append(model.iaip.glasses_shared)
        params.append(model.iaip.age_shared)
        params.append(model.iaip.smile_shared)
        all_l1 = []
        for y in range(4):
            all_linear1_params = torch.cat([x.view(-1) for x in params[y]])
            all_l1.append(args.lambda_sparsity * torch.sum(torch.abs(all_linear1_params)).unsqueeze(0))
        l1_loss = torch.mean(torch.cat(all_l1))
        all_losses.l1 = l1_loss.item()
        g_loss += (l1_loss)


    # direction loss
    mul = torch.cat(mul, dim =0)
    shared_params = [torch.tile(model.iaip.gender_shared, (args.batch_size, 18, 1)), 
                     torch.tile(model.iaip.glasses_shared, (args.batch_size, 18, 1)),
                     torch.tile(model.iaip.age_shared, (args.batch_size, 18, 1)), 
                     torch.tile(model.iaip.smile_shared, (args.batch_size, 18, 1))]
    shared_params = torch.cat(shared_params, dim=0)
    criterion = nn.CosineSimilarity(dim=2)
    loss_dir = 1 - torch.mean(torch.abs(criterion(mul, shared_params)))
    all_losses.direction = loss_dir.item()
    g_loss += (loss_dir)

    # cosine similarity loss
    if args.lambda_cos > 0:
        loss_cosine = frs.cosine_loss(img_src, imgs_trg[attr_idx]) * args.lambda_cos
        all_losses.cosine = loss_cosine.item()
        g_loss += loss_cosine
 
    # neighborhood loss
    if args.lambda_nb > 0:
        nb_losses = []
        for a in range(4):
            nb_losses.append(torch.mean(torch.norm(lat - outputs['all'][a], dim=(1, 2))).unsqueeze(0))
        loss_nb = torch.mean(torch.cat(nb_losses))  * args.lambda_nb
        all_losses.nb = loss_nb.item()
        g_loss += loss_nb

    return g_loss, all_losses


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = param.data


def build_optim_schedulers(args, model):
    if args.optim == 'Adam':
        optim = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, betas=(0.98, 0.98),
            weight_decay=args.weight_decay
        )

    elif args.optim == 'AdaBelief':
        optim = AdaBelief(
            model.parameters(),
            lr=args.lr, eps=1e-16, betas=(0.98, 0.98), weight_decay=args.weight_decay, weight_decouple=True, rectify=False,
            amsgrad=False, fixed_decay=False, print_change_log=False)

    elif args.optim == 'RMSprop':
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay
        )
    else:
        raise 'unknown optimizer'

    scheduler = CosineAnnealingWarmupRestarts(optim,
                                        first_cycle_steps=args.total_iters,
                                        cycle_mult=1.0,
                                        max_lr=1e-3,
                                        min_lr=1e-8,
                                        warmup_steps=args.warmup_steps,
                                        gamma=1.0)

    return optim, scheduler


def save_checkpoint(model, checkpoint_dir, step, suffix="nets"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    fname = os.path.join(checkpoint_dir, "{:06d}.ckpt".format(step))
    print('Saving checkpoint into %s...' % fname)
    torch.save(model.state_dict(), fname)


def load_checkpoint(model, checkpoint_dir, resume_iter=-1):
    if resume_iter > 0:
        model_path = os.path.join(checkpoint_dir, "{:06d}.ckpt".format(resume_iter))
        ckpt = torch.load(model_path, map_location='cuda')
        model.load_state_dict(ckpt)
        print("Loading checkpoint from {}...".format(model_path))


def load_id_model(args, device):
    frs = IDLoss().to(device) if args.lambda_cos > 0 else None
    return frs


def map_to_cuda(model, device):
    for _, module in model.items():
        module.to(device)


def build_stylegan_model_g(args):
    model = StyleGAN2Generator(1024, 512, 8)
    model._load_pretrain(args.stylegan2_path)
    return model


def run(args, device):

    #create models
    model, attr_classifier = build_model(args, device)

    # print number of trainale parameters
    utils.print_network(model, 'ID-Style')

    # resume training if necessary
    load_checkpoint(model, args.checkpoint_dir, args.resume_iter)

    # build optimizers and schedulers
    optim, scheduler = build_optim_schedulers(args, model)

    # build id net
    frs = load_id_model(args, device)

    # build stylegan's generator
    gan_model = build_stylegan_model_g(args)
    gan_model.eval().to(device)
    for param in gan_model.parameters():
        param.requires_grad = False

    # build dataloaders
    loaders = Munch(src=get_data_loader(source_path=args.source_path,
                                        attr_path=args.attr_path,
                                        latent_path=args.latent_path,
                                        dataset_name='ffhq',
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        mode="train",
                                        latent_type=args.latent_type),
                    val=get_data_loader(source_path=args.source_path,
                                        attr_path=args.attr_path,
                                        latent_path=args.latent_path,
                                        dataset_name='ffhq',
                                        batch_size=8,
                                        num_workers=args.num_workers,
                                        mode="test",
                                        latent_type=args.latent_type))
    
    # fetch random validation images for visual validation
    fetcher = InputFetcher(loaders.src, device=device)
    fetcher_val = InputFetcher(loaders.val, device=device)
    inputs_val = next(fetcher_val)
    
    
    start_time = time.time()
    current_time = time.asctime(time.localtime(time.time()))

    # write in tensorboard
    writer = SummaryWriter('./results/' + str(current_time) + '/tensorboard-logs/train')

    # create a backup of current run
    utils.save_config(args, current_time)

    # logging
    handlers = [logging.FileHandler('./results/' + str(current_time) + '/log.txt'), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers)

    all_losses = dict()
    flag = False

    # start training
    print('Start training...')
    for i in range(args.resume_iter, args.total_iters):
        inputs = next(fetcher)
        model.train()
        img_src, lat, lab_src = inputs.img_src, inputs.lat, inputs.lab_src

        lab_trg = labels_target_stage1(lab_src)
        g_loss, g_losses = compute_g_loss(i, model, attr_classifier, gan_model, args, img_src, lat, lab_src, lab_trg,
                                            frs=frs, iter=i)
        g_loss = g_loss
        g_loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
       
        # make a dictionary of losses
        for key, value in g_losses.items():
            if flag:
                all_losses['loss_' + key] += value / args.print_every
            else:
                all_losses['loss_' + key] = value / args.print_every
        flag = True

        # save model checkpoints
        if  (i + 1) % args.save_every == 0:
            save_checkpoint(model, os.path.join(args.checkpoint_dir, str(current_time), 'checkpoints'),
                            i + 1, "nets")

        # print out log info
        if ((i + 1) % args.print_every == 0):
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)

            all_losses["lr"] = optim.param_groups[0]['lr']
            log += ' '.join(
                ['%s: [%.4f]' % (key, value) if 'lr' not in key else '%s: [%.6f]' % (key, value) for key, value in
                 all_losses.items()])


            logging.info(log)
            logging.info('*'*100)

            # write in tensorboard
            print('write in tensorboard')
            info = {
                'epoch': i+1,
                'all_losses': all_losses
            }
            write_in_tensorboard(writer, info)
            all_losses = dict()
            flag = False

        # visual test
        if (i + 1) % args.sample_every == 0:
            os.makedirs(args.sample_dir, exist_ok=True)
            model.eval()
            with torch.no_grad():
                utils.debug_image(model, gan_model, args, current_time, inputs=inputs_val, step=i + 1, device=device, iter=i)

def load_labels(path):
    file1 = open(path, 'r')
    labels = []
    lines = file1.readlines()
    for i, line in enumerate(lines):
        if i > 1:
            line_content = str.split(line.strip(), '\t')
            label = [int(line_content[1]), int(line_content[2]), int(line_content[3]), int(line_content[4])]
            labels.append(label)
    return np.array(labels)


def visual_test(args, device):
    src_latents_path = args.latent_path
    save_dir = args.sample_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = IDStyle().to(device)
    load_checkpoint(model, args.checkpoint_dir, args.resume_iter)
    
    gan_model = build_stylegan_model_g(args)
    gan_model.eval().to('cuda')

    attributes = [0, 1, 2, 3]

    all_latents = np.load(src_latents_path)
    src_labels = load_labels(path=args.attr_path)
    src_labels = torch.from_numpy(src_labels).float().cuda()
    
    for i, latent in enumerate(all_latents):
        latent_code = torch.from_numpy(latent).float().cuda()
        lat, lab_src = latent_code, src_labels[i]
        lab_trg_pos = labels_target_stage1(lab_src).unsqueeze(0)

        lats_pos, _ = model(lat, lab_trg_pos)
        for attr_idx in attributes:
            lat_pos = lats_pos[attr_idx]
            if lat_pos.shape[1] != 18:
                lat_pos = torch.tile(lat_pos, (1, 18, 1))
            
            fake_pos = gan_model(lat_pos)
            fake_pos = torch.clamp(fake_pos * 0.5 + 0.5, 0, 1)
            vutils.save_image(fake_pos.data, os.path.join(save_dir,
                                                          f'{i:06d}_{attr_idx}_{1 - lab_src[attr_idx]}.png'), padding=0)
    print('done!')