import torch.nn as nn
from .utils import create_mlp

class MultiLayerPerceptron_Torch(nn.Module):

    def __init__(self, input_dim, output_dim = 1, layers = [], 
        activation = 'ReLU', dropout = 0):
        super(MultiLayerPerceptron_Torch, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = create_mlp(input_dim, layers + [output_dim], activation, dropout)

    def forward(self, x):
        return self.mlp(x)

    def get_last_weights(self):
        return self.mlp[-2].weight
    
    def replace_last_weights(self, weight):
        # This is a workaround as the current library does not allow direct computation
        del self.mlp[-2].weight
        self.mlp[-2].weight = weight
        return self
