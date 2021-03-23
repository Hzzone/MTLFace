import torch
from common.ops import convert_to_cuda


class DataPrefetcher(object):
    def __init__(self, loader):
        self.__loader__ = loader
        self.loader = iter(self.__loader__)
        # eg. lambda x: x.sub_(0.5).div_(0.5)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            next_input = convert_to_cuda(self.next_input)
            self.next_input = next_input

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input
