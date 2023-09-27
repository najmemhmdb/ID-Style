import os
import argparse
from munch import Munch
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import torch
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
from core.solver import run, visual_test
from core.model import IDStyle
from core.data_loader import (
    get_data_loader,
    InputFetcher
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--latent_type', type=str, default='wp', help='[w, wp]')

    parser.add_argument('--lambda_cls', type=float, default=2,
                        help='Weight for classification')
    
    parser.add_argument('--lambda_cos', type=float, default=5,
                         help='Weight for cosine identity constraint')

    parser.add_argument('--lambda_nb', type=float, default=0.3,
                        help='Weight of neighbouring constraint')

    parser.add_argument('--lambda_sparsity', type=float, default=1,
                        help='Weight of sparsity constraint')
    
    parser.add_argument('--total_iters', type=int, default=50000,
                        help='Number of total iterations')

    
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help="Number of warmup steps")

    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')

    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')

    parser.add_argument('--optim', type=str, default='AdaBelief',
                        help='Adam, AdaBelief, or RMSprop')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for')

    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
          
    parser.add_argument('--mode', type=str,
                        choices=['train', "test"],
                        default='train',
                        help='This argument is used in solver')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers used in DataLoader')

    parser.add_argument('--stylegan2_path', type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt")

    parser.add_argument('--source_path', type=str, default='./dataset/')
    parser.add_argument('--attr_path', type=str, default='./attributes/')
    parser.add_argument('--latent_path', type=str, default='./dataset/')
    parser.add_argument('--sample_dir', type=str,
                        default='./results/',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/',
                        help='Directory for saving network checkpoints')

    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--sample_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)

    args = parser.parse_args()

    return args


def main(args): 
    device = torch.device('cuda')
    # fix random seeds
    fix_seed = 1234
    random.seed(777)
    torch.manual_seed(fix_seed)
    torch.random.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.mode == 'train':
        run(args, device)
    if args.mode == 'test':
        visual_test(args, device)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

