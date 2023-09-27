import math
import copy
import os
import numpy as np
from munch import Munch
import gc
import torch
from torch import nn
import torch.nn.functional as F

class IAIP(nn.Module):
    def __init__(self):
        super(IAIP, self).__init__()
        
        self.fc = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.ReLU()
            )
       
        self.gender_shared = torch.nn.parameter.Parameter(torch.normal(0, 0.1, (1,512)), requires_grad=True) 
        self.glasses_shared = torch.nn.parameter.Parameter(torch.normal(0, 0.1, (1,512)), requires_grad=True) 
        self.age_shared =  torch.nn.parameter.Parameter(torch.normal(0, 0.1, (1,512)), requires_grad=True) 
        self.smile_shared = torch.nn.parameter.Parameter(torch.normal(0, 0.1, (1,512)), requires_grad=True) 
        self.l18 = nn.Linear(18, 18)
        self.relu18 = nn.ReLU()
        self.l1024 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
      
    def forward(self, w, d, iter):
        h = w
        specific_delta = self.fc(h).permute(0, 2, 1)
        specific_delta = self.l18(specific_delta)
        specific_delta = self.relu18(specific_delta).permute(0, 2, 1)
        specific_delta = self.l1024(specific_delta)

        return specific_delta


class IDStyle(nn.Module):
    def __init__(self):
        super(IDStyle, self).__init__()
        self.iaip = IAIP()
        self.embedding_attr = nn.Embedding(4, 18, max_norm=None)
        
        self.embedding_attr.weight.data = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0., 0, 0, 0, 0, 0, 0, 0, 0],
                                      [ 0,  0, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [ 0,  0,  0,  0, 1,1,1,1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                      [ 0,  0,  0,  0, 1,1, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
        for param in self.embedding_attr.parameters():
            param.requires_grad = False

        self.attribute_idx = torch.LongTensor(list(range(4)))

        self.pe = torch.zeros(18, 512).float()
        self.pe.require_grad = False
        position = torch.arange(0, 18).float().unsqueeze(1)
        div_term = (torch.arange(0, 512, 2).float() * -(math.log(10000.0) / 512)).exp()
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)


    def forward(self, w_org, d, iter=0):
        embd_w = torch.tile(self.pe[:, :w_org.size(1)].cuda(), (w_org.shape[0], 1, 1))
        w_org2 = torch.cat([w_org, embd_w], dim=2)
        specific_delta = self.iaip(w_org2, d, iter)
        d_prim = d.clone()
        dir_gender = d_prim[:, 0]
        dir_glasses = d_prim[:, 1]
        dir_age = d_prim[:, 2]
        dir_smile = d_prim[:, 3]

        gender_res =  self.iaip.gender_shared / torch.norm(self.iaip.gender_shared, p=1) *  specific_delta
        glasses_res = self.iaip.glasses_shared / torch.norm(self.iaip.glasses_shared, p=1) * specific_delta
        age_res = self.iaip.age_shared / torch.norm(self.iaip.age_shared, p=1) * specific_delta
        smile_res = self.iaip.smile_shared / torch.norm(self.iaip.smile_shared, p=1) * specific_delta

        gender_res_p = (self.iaip.gender_shared / torch.norm(self.iaip.gender_shared, p=1)) + gender_res
        glasses_res_p = (self.iaip.glasses_shared / torch.norm(self.iaip.glasses_shared, p=1)) + glasses_res
        age_res_p = (self.iaip.age_shared / torch.norm(self.iaip.age_shared, p=1)) + age_res
        smile_res_p = (self.iaip.smile_shared / torch.norm(self.iaip.smile_shared, p=1)) + smile_res
        delta_gender = torch.einsum("b, bjk -> bjk", dir_gender, gender_res_p)
        delta_glasses = torch.einsum("b, bjk -> bjk", dir_glasses, glasses_res_p)
        delta_age = torch.einsum("b, bjk -> bjk", dir_age, age_res_p)
        delta_smile = torch.einsum("b, bjk -> bjk", dir_smile, smile_res_p)
        
        embd = torch.relu(self.embedding_attr(self.attribute_idx.cuda()))
        delta_gender_emb = torch.einsum("j, bjk -> bjk", embd[0], delta_gender)
        delta_glasses_emb = torch.einsum("j, bjk -> bjk", embd[1], delta_glasses)
        delta_age_emb = torch.einsum("j, bjk -> bjk", embd[2], delta_age)
        delta_smile_emb = torch.einsum("j, bjk -> bjk", embd[3], delta_smile)

        w_new_gender = delta_gender_emb + w_org
        w_new_glasses = delta_glasses_emb + w_org
        w_new_age = delta_age_emb + w_org
        w_new_smile = delta_smile_emb + w_org
        
        return [w_new_gender, w_new_glasses, w_new_age, w_new_smile], [gender_res, glasses_res, age_res, smile_res]


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, upsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=4, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = nn.Sequential(*blocks)
        self.out_cls = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, num_domains, 1, 1, 0))

        self.out_src = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, 1, 1, 1, 0))

    def forward(self, x, y=None):
        out = self.main(x)
        out_cls = self.out_cls(out).view(x.size(0), -1)  # (batch, num_domains)
        out_src = self.out_src(out).view(x.size(0))
        return out_cls, out_src


def build_model(args, device=None):
    model = IDStyle().to(device)

    attr_classifier = Discriminator(
        img_size=256,
        num_domains=4
    ).to(device)
    attr_classifier.load_state_dict(torch.load('./pretrained_models/050000_nets_ema.ckpt', map_location='cuda')['discriminator'])
    
    attr_classifier.eval()
    for param in attr_classifier.parameters():
        param.requires_grad = False
        
    return model, attr_classifier

