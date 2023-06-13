from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Dropout
from torch import softmax
from torch.nn.init import kaiming_normal_

class CNN(Module):   
    def __init__(self, n_dim, linear_dim):
        super(CNN, self).__init__()
        self.n_dim = n_dim
        self.linear_dim = linear_dim

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=4, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(4, 8, kernel_size=4, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2),
            # Defining another 2D convolution layer
            Conv2d(8, 16, kernel_size=4, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(16, 32, kernel_size=4, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )
        
        self.linear_layers = Sequential(
            Linear(self.linear_dim, 2, bias=False)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = softmax(self.linear_layers(x), dim=1)
        return x

class LLR(Module):   
    def __init__(self, n_dim):
        super(LLR, self).__init__()
        self.linear = Linear(n_dim, 2, bias=False)

    # Defining the forward pass    
    def forward(self, x):
        x = softmax(self.linear(x), dim=1)
        return x

class MLP(Module):   
    def __init__(self, n_dim):
        super(MLP, self).__init__()
        self.n_dim = n_dim

        self.linear_layers = Sequential(
            Linear(self.n_dim, int(256),  bias=False),
            ReLU(),
            Dropout(p=0.3),
            Linear(int(256), int(256),  bias=False),
            ReLU(),
            Dropout(p=0.3),
            Linear(int(256), int(64),  bias=False),
            ReLU(),
            Dropout(p=0.3),
            Linear(int(64),2, bias=False)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = softmax(self.linear_layers(x), dim=1)
        return x
    
def init_he_normal(layer):
    if isinstance(layer, Conv2d) or isinstance(layer,Linear):
        kaiming_normal_(layer.weight)

models_dict = {
    "CNN": CNN,
    "LLR": LLR,
    "MLP": MLP
}
