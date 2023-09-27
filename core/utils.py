import os
from os.path import join as ospj
import json
import glob
import shutil
import yaml
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import logging

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def labels_target_stage1(labels_src):
    labels_trg = torch.ones(labels_src.shape).cuda()
    labels_trg -= labels_src
    labels_trg = labels_trg * 2 - 1
    return labels_trg


def toggle_lab(labels_src, attr_idx):

    labels_trg = labels_src.clone()

    for label_trg in labels_trg:
        if label_trg[attr_idx] == 0: 
            label_trg[attr_idx] = 1.0
        else: 
            label_trg[attr_idx] = 0.0
    return labels_trg



def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_reconstruction(model, model_gan, args, img_src, lat, filename, lab_src, iter):
    N = img_src.size(0)
    out_concat = [img_src]
    lab_trg = labels_target_stage1(lab_src)
    lats_out, _ = model(lat, lab_trg, iter)
    for s in range(4):
        if lats_out[s].shape[1] != 18:
            lat_out = torch.tile(lats_out[s], (1, 18, 1))
        else: 
            lat_out = lats_out[s]
        img_trg = model_gan(lat_out)
        out_concat.append(img_trg)
    out_concat = torch.cat(out_concat, dim=2)
    save_image(out_concat, N, filename)
    del out_concat   


@torch.no_grad()
def debug_image(model, model_gan, args, current_time, inputs, step, device, iter):
    img_src, lat, lab_src = inputs.img_src, inputs.lat, inputs.lab_src
    os.makedirs(os.path.join(args.sample_dir, str(current_time), 'samples'), exist_ok=True)
    filename = ospj(os.path.join(args.sample_dir, str(current_time), 'samples'), '%06d_rec.jpg' % (step))
    translate_reconstruction(model, model_gan, args, img_src, lat, filename, lab_src, iter)


def write_in_tensorboard(writer, info):
    for key, value in info.get('all_losses').items():
        writer.add_scalar(key, value, info.get('epoch'))


def save_config(args, current_time):
    filename = './results/' + str(current_time) + '/config.yaml'
    shutil.copy('./core/model.py', './results/' + str(current_time) + '/model.py')
    shutil.copy('./core/solver.py', './results/' + str(current_time) + '/solver.py')
    shutil.copy('./train.sh', './results/' + str(current_time) + '/train.sh')
    with open(filename, 'w') as yaml_file:
        yaml.dump(args, yaml_file, default_flow_style=False)
