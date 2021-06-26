import torch
from mmcv.ops import ModulatedDeformConv2dPack as DCNv2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = torch.arange(45).reshape(1, 1, 5, 9).float().to(device)
m = torch.nn.Conv2d(1, 1, 3, 1, 1).to(device)
torch.nn.init.constant_(m.weight, 1.0)
torch.nn.init.constant_(m.bias, 0)
model = DCNv2(in_channels=1,
              out_channels=1,
              kernel_size=3,
              stride=1,
              padding=1,
              deform_groups=1,
              bias=False).to(device)
torch.nn.init.constant_(model.weight, 1.0)
output = model(x)
print(output.squeeze())
output = m(x)
print(output.squeeze())
