from torch import nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    """
    ResNet block for CIFAR10 dataset
    """
    def __init__(self, f_in, f_out, downsample=False):
        super(ResNetBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f_out)

        if downsample or f_in != f_out:
            self.skip = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out)
            )
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return F.relu(out)


class ResNetCifar(nn.Module):
    """
    A residual neural network for CIFAR-10.
    """
    def __init__(self, plan, initializer, outputs=10):
        """
        Params
        ------
        - plan [list] : structure of network modules [(W, D), (2*W, D), (4*W, D)].
          W - filters, D - number of ResNet blocks.
        - initializer [function] : initializer apply to all layers with
          initialization purpose
        - outputs [int] : amount of output classes
        """
        super(ResNetCifar, self).__init__()
        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNetBlock(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
