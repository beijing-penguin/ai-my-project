# x = self.rpn(x)
# print(x.shape)   torch.Size([1, 512, 128, 128])
# x1 = x.permute(0,2,3,1).contiguous()  # channels last
# print(x.shape)
import numpy as np
import torch.nn as nn
import torch
print(torch.__version__)
class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
x = torch.randn(1, 512, 3, 4)
print(x.shape)
#test = BasicConv(512, 10*2, 3, 1, relu=False,bn=False)
#print(test(x))
x1 = x.permute(0,2,3,1).contiguous()
print(x1.shape) 
print(x1)
b = x1.size()  # batch_size, h, w, c
print(b)
x1 = x1.view(b[0]*b[1], b[2], b[3])
print(x1.shape)


x2 ,(final_hidden_state, final_cell_state)= torch.nn.GRU(512,128, bidirectional=True, batch_first=True)(x1)
print(final_hidden_state.shape)
print(x2.shape)
#x2 = x2.permute(1, 0, 2)
n_hidden = 10
#hidden = final_cell_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
attn_weights = torch.bmm(x2, final_cell_state).squeeze(2) # attn_weights : [batch_size, n_step]
soft_attn_weights = F.softmax(attn_weights, 1)
        
context = torch.bmm(x2, soft_attn_weights.unsqueeze(2)).squeeze(2)


xsz = x.size()
x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)
print(x3.shape)

x3 = x3.permute(0,3,1,2).contiguous()
print(x3.shape)
lstm_fc = BasicConv(256, 512,1,1,relu=True, bn=False)
x3 = lstm_fc(x3)
print(x3.shape)
x = x3

rpn_class = BasicConv(512, 10*2, 1, 1, relu=False,bn=False)

cls = rpn_class(x)
print(cls.shape)
rpn_regress = BasicConv(512, 10 * 2, 1, 1, relu=False, bn=False)
regr = rpn_regress(x)
print(regr.shape)
cls = cls.permute(0,2,3,1).contiguous()
print(cls.shape)
regr = regr.permute(0,2,3,1).contiguous()
print(regr.shape)
cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)
regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2)
print(cls.shape)
print(regr.shape)
#print(cls)
print(cls)
print(regr)

print(np.minimum([-0.5,-0.5,-0.5 , 15.5 ,15.5 ,15.5], 32))
print(np.maximum([1,2,3,4,5],2))
#y = cls[0][0]
#print(y)
#print(y != -1)
#cls_keep = (y != -1).nonzero()[:, 0]
#print(cls_keep)