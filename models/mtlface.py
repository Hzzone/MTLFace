import argparse
import tqdm

from .fr import FR
from .fas import FAS
from common.ops import load_network

'''
python -m torch.distributed.launch --nproc_per_node=8 --master_port=17647 main.py \
    --train_fr --backbone_name ir50 --head_s 64 --head_m 0.35 \
    --weight_decay 5e-4 --momentum 0.9 --fr_age_loss_weight 0.001 --fr_da_loss_weight 0.002 --age_group 7 \
    --gamma 0.1 --milestone 20000 23000 --warmup 1000 --learning_rate 0.1 \
    --dataset_name scaf --image_size 112 --num_iter 36000 --batch_size 64 --amp
    
python -m torch.distributed.launch --nproc_per_node=8 --master_port=17647 main.py \
    --train_fas --backbone_name ir50 --age_group 7 \
    --dataset_name scaf --image_size 112 --num_iter 36000 --batch_size 64 \
    --d_lr 1e-4 --g_lr 1e-4 --fas_gan_loss_weight 75 --fas_age_loss_weight 10 --fas_id_loss_weight 0.002
'''


class MTLFace(object):

    def __init__(self, opt):
        self.opt = opt
        self.fr = FR(opt)
        self.fr.set_loader()
        self.fr.set_model()
        if opt.train_fas:
            if opt.id_pretrained_path is not None:
                self.fr.backbone.load_state_dict(load_network(opt.id_pretrained_path))
            if opt.age_pretrained_path is not None:
                self.fr.estimation_network.load_state_dict(load_network(opt.age_pretrained_path))
            self.fas = FAS(opt)
            self.fas.set_loader()
            self.fas.set_model()

    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()

        parser.add_argument("--train_fr", help='train_fr', action='store_true')
        parser.add_argument("--train_fas", help='train_fas', action='store_true')

        # BACKBONE, HEAD
        parser.add_argument("--backbone_name", help='backbone', type=str)
        parser.add_argument("--head_s", help='s of cosface or arcface', type=float, default=64)
        parser.add_argument("--head_m", help='m of cosface or arcface', type=float, default=0.35)

        # OPTIMIZED
        parser.add_argument("--weight_decay", help='weight-decay', type=float, default=5e-4)
        parser.add_argument("--momentum", help='momentum', type=float, default=0.9)

        # LOSS
        parser.add_argument("--fr_age_loss_weight", help='age loss weight', type=float, default=0.0)
        parser.add_argument("--fr_da_loss_weight", help='cross age domain adaption loss weight', type=float,
                            default=0.0)
        parser.add_argument("--age_group", help='age_group', default=7, type=int)

        # LR
        parser.add_argument("--gamma", help='learning-rate gamma', type=float, default=0.1)
        parser.add_argument("--milestone", help='milestones', type=int, nargs='*', default=[20, 40, 60])
        parser.add_argument("--warmup", help='learning rate warmup epoch', type=int, default=5)
        parser.add_argument("--learning_rate", help='learning-rate', type=float, default=0.1)

        # TRAINING
        parser.add_argument("--dataset_name", "-d", help='input image size', type=str)
        parser.add_argument("--image_size", help='input image size', default=224, type=int)
        parser.add_argument("--num_iter", help='total epochs', type=int, default=125)
        parser.add_argument("--restore_iter", help='restore_iter', default=0, type=int)
        parser.add_argument("--batch_size", help='batch-size', default=0, type=int)
        parser.add_argument("--val_interval", help='val dataset interval iteration', type=int, default=1000)

        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument("--num_worker", help='dataloader num-worker', default=32, type=int)
        parser.add_argument("--local_rank", help='local process rank, not need to be set.', default=0, type=int)

        parser.add_argument("--amp", help='amp', action='store_true')

        # FAS
        parser.add_argument("--d_lr", help='learning-rate', type=float, default=1e-4)
        parser.add_argument("--g_lr", help='learning-rate', type=float, default=1e-4)
        parser.add_argument("--fas_gan_loss_weight", help='gan_loss_weight', type=float)
        parser.add_argument("--fas_id_loss_weight", help='id_loss_weight', type=float)
        parser.add_argument("--fas_age_loss_weight", help='age_loss_weight', type=float)
        parser.add_argument("--id_pretrained_path", help='id_pretrained_path', type=str)
        parser.add_argument("--age_pretrained_path", help='age_pretrained_path', type=str)

        return parser

    def fit(self):
        opt = self.opt
        # training routine
        for n_iter in tqdm.trange(opt.restore_iter + 1, opt.num_iter + 1, disable=(opt.local_rank != 0)):
            # img, label, age, gender
            fr_inputs = self.fr.prefetcher.next()
            if opt.train_fr:
                self.fr.train(fr_inputs, n_iter)
            if opt.train_fas:
                # target_img, target_label
                fas_inputs = self.fas.prefetcher.next()
                # backbone, age_estimation, source_img, target_img, source_label, target_label
                # You can also use other attributes for aligning
                _fas_inputs = [self.fr.backbone.module, self.fr.estimation_network,
                               fr_inputs[0], fas_inputs[0], fr_inputs[1], fas_inputs[1]]
                self.fas.train(_fas_inputs, n_iter)
            if n_iter % opt.val_interval == 0:
                if opt.train_fr:
                    self.fr.validate(n_iter)
                if opt.train_fas:
                    self.fas.validate(n_iter)
