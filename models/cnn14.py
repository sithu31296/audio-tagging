import torch
from torch import nn, Tensor


class Conv2Layer(nn.Module):
    def __init__(self, c1, c2, pool_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(pool_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.dropout(x)
        return x



class CNN14(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = Conv2Layer(1, 64)
        self.conv_block2 = Conv2Layer(64, 128)
        self.conv_block3 = Conv2Layer(128, 256)
        self.conv_block4 = Conv2Layer(256, 512)
        self.conv_block5 = Conv2Layer(512, 1024)
        self.conv_block6 = Conv2Layer(1024, 2048, 1)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(2048, 2048)
        self.fc = nn.Linear(2048, num_classes)

    def _init_weights(self, pretrained: str = None):
        if pretrained:
            print(f"Loading Pretrained Weights from {pretrained}")
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            model_dict = self.state_dict()
            for k in model_dict.keys():
                if not k.startswith('fc.'):
                    model_dict[k] = pretrained_dict[k]
            self.load_state_dict(model_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        # x [B, 1, mel_bins, time_steps]
        x = x.permute(0, 2, 3, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = x.mean(3)
        x = x.max(dim=2)[0] + x.mean(2)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc(x)


class CNN14DecisionLevelMax(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.interpolate_ratio = 32     # downsampled ratio

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = Conv2Layer(1, 64)
        self.conv_block2 = Conv2Layer(64, 128)
        self.conv_block3 = Conv2Layer(128, 256)
        self.conv_block4 = Conv2Layer(256, 512)
        self.conv_block5 = Conv2Layer(512, 1024)
        self.conv_block6 = Conv2Layer(1024, 2048, 1)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, 1, 1)
        self.avgpool = nn.AvgPool1d(3, 1, 1)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc = nn.Linear(2048, num_classes)

    def _init_weights(self, pretrained: str = None):
        if pretrained:
            print(f"Loading Pretrained Weights from {pretrained}")
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            model_dict = self.state_dict()
            for k in model_dict.keys():
                if not k.startswith('fc.'):
                    model_dict[k] = pretrained_dict[k]
            self.load_state_dict(model_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        # x [B, 1, mel_bins, time_steps]
        num_frames = x.shape[-1]
        x = x.permute(0, 2, 3, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = x.mean(3)
        x = self.maxpool(x) + self.avgpool(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.relu(self.fc1(x))

        segmentwise = self.fc(x).sigmoid()
        clipwise = segmentwise.max(dim=1)[0]

        # get framewise output
        framewise = interpolate(segmentwise, self.interpolate_ratio)
        framewise = pad_framewise(framewise, num_frames)

        return framewise, clipwise



def interpolate(x, ratio):
    B, T, C = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(B, T*ratio, C)
    return upsampled


def pad_framewise(framewise, num_frames):
    pad = framewise[:, -1:, :].repeat(1, num_frames-framewise.shape[1], 1)
    return torch.cat([framewise, pad], dim=1)    

    
if __name__ == '__main__':
    import time
    model = CNN14(527)
    model.load_state_dict(torch.load('checkpoints/cnn14.pth', map_location='cpu'))
    x = torch.randn(3, 1, 64, 701)
    start = time.time()
    y = model(x)
    print(time.time()-start)
    print(y.shape)
    print(y.min(), y.max())