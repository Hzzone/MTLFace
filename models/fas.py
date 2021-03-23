import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.cuda.amp as amp

from common.sampler import RandomSampler
from common.data_prefetcher import DataPrefetcher
from common.ops import convert_to_ddp
from . import BasicTask
from common.dataset import AgingDataset


class FAS(BasicTask):

    def set_loader(self):
        opt = self.opt

        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize([opt.image_size, opt.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, ], std=[0.5, ])
            ])
        train_dataset = AgingDataset(opt.dataset_name, age_group=opt.age_group,
                                     total_pairs=opt.num_iter * opt.batch_size, transforms=train_transform)

        weights = None
        sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                num_iter=opt.num_iter, restore_iter=opt.restore_iter, weights=weights)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, sampler=sampler, pin_memory=True,
            num_workers=opt.num_worker, drop_last=True
        )
        self.prefetcher = DataPrefetcher(train_loader)

    def set_model(self):
        opt = self.opt
        from common.networks import AgingModule, PatchDiscriminator
        generator = AgingModule(age_group=opt.age_group)
        discriminator = PatchDiscriminator(opt.age_group, norm_layer='sn', repeat_num=4)

        d_optim = torch.optim.Adam(discriminator.parameters(), opt.d_lr, betas=(0.5, 0.99))
        g_optim = torch.optim.Adam(generator.parameters(), opt.g_lr, betas=(0.5, 0.99))

        generator, discriminator = convert_to_ddp(generator, discriminator)

        scaler = amp.GradScaler()
        self.generator = generator
        self.discriminator = discriminator
        self.d_optim = d_optim
        self.g_optim = g_optim
        self.scaler = scaler

        self.logger.modules = [generator, discriminator, d_optim, g_optim, scaler]
        if opt.restore_iter > 0:
            self.logger.load_checkpoints(opt.restore_iter)

    def validate(self, n_iter):
        pass

    def train(self, inputs, n_iter):
        opt = self.opt

        backbone, age_estimation, source_img, target_img, source_label, target_label = inputs
        backbone.eval()
        self.generator.train()
        self.discriminator.train()

        ###################################D loss###############################

        self.d_optim.zero_grad()

        with torch.no_grad():
            with amp.autocast(enabled=opt.amp):
                x_1, x_2, x_3, x_4, x_5, x_id, x_age = backbone(source_img, return_shortcuts=True)
            x_1, x_2, x_3, x_4, x_5, x_id, x_age = \
                x_1.float(), x_2.float(), x_3.float(), x_4.float(), x_5.float(), x_id.float(), x_age.float()

        g_source = self.generator(source_img, x_1, x_2, x_3, x_4, x_5, x_id, x_age, condition=target_label)

        d1_logit = self.discriminator(target_img, target_label)
        d3_logit = self.discriminator(g_source.detach().float(), target_label)
        d_loss = 0.5 * (torch.mean((d1_logit - 1) ** 2) + torch.mean(d3_logit ** 2))
        d_loss.backward()
        self.d_optim.step()

        with amp.autocast(enabled=opt.amp):
            _, g_x_id, g_x_age = backbone(g_source, return_age=True)
        g_x_id, g_x_age = g_x_id.float(), g_x_age.float()

        ###################################gan_loss###############################
        self.g_optim.zero_grad()
        g_logit = self.discriminator(g_source, target_label)
        g_loss = 0.5 * torch.mean((g_logit - 1) ** 2)

        ################################id_loss#############################
        fas_id_loss = F.mse_loss(x_id, g_x_id)

        ################################age_loss#############################
        fas_age_loss = F.cross_entropy(age_estimation(g_x_age)[1], target_label)

        total_loss = g_loss * opt.fas_gan_loss_weight \
                     + fas_id_loss * opt.fas_id_loss_weight \
                     + fas_age_loss * opt.fas_age_loss_weight

        if opt.amp:
            total_loss = self.scaler.scale(total_loss)
        total_loss.backward()
        if opt.amp:
            self.scaler.step(self.g_optim)
            self.scaler.update()
        else:
            self.g_optim.step()

        self.logger.msg([d1_logit, d3_logit, g_logit, fas_id_loss, fas_age_loss], n_iter)
