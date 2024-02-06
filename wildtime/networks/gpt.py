import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2Generator(nn.Module):
    def __init__(self):
        super(GPT2Generator, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        # activator = nn.Sigmoid()

    def forward(self, x, labels):
        return self.model(x, labels=labels)
