from mxnet import nd
from mxnet.gluon import nn


class Model(nn.Block):
    def __init__(self):
        super(Model, self).__init__()

        self.net = nn.Sequential()
        self.net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.MaxPool2D(pool_size=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2),
                nn.Conv2D(channels=32, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu", flatten=True),
                nn.Dense(84, activation="relu"),
                nn.Dense(20))

    def forward(self, x):

        x = self.net(x)
        return x
