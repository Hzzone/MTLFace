import os.path as osp
from common.ops import LoggerX

class BasicTask(object):
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root='../output')

    def set_loader(self):
        pass

    def set_model(self):
        pass

    def validate(self, n_iter):
        pass

    def adjust_learning_rate(self, step):
        pass

    def train(self, inputs, n_iter):
        pass
