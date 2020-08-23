import torch
from torch import nn
from torch import functional as F
from torchvision.models import vgg19
from torch.autograd import Variable
import settings
class SEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        mid = int(input_dim /4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class NoSEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

    def forward(self, x):
        return x

SE = SEBlock if settings.use_se else NoSEBlock
class Res(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Res, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        out=x+self.conv(x)
        return out
class Tradition(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Tradition, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2))
    def forward(self, x):
        out=self.conv(x)
        return out
class Dilation(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Dilation, self).__init__()
        self.path1=nn.Sequential(nn.Conv2d(in_channel, out_channel,3,1,1,1),nn.LeakyReLU(0.2))
        self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3), nn.LeakyReLU(0.2))
        self.path5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), nn.LeakyReLU(0.2))
    def forward(self, x):
        path1 = self.path1(x)
        path3 = self.path3(x)
        path5 = self.path5(x)
        out=path1+path3+path5
        return out
class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim):
        super().__init__()
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, 3,1,1)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, 3,1,1)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, 3,1,1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.se = SE(oup_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = self.sigmoid(self.conv_xz(x))
        f = self.tanh(self.conv_xn(x))
        h = z * f
        h = self.relu(self.se(h))
        return h
class Compact(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Compact, self).__init__()
        self.df1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1,1),nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.2))
        self.conv1_1 = nn.Conv2d(2*out_channel, out_channel, 1, 1)
        self.conv1_2 = nn.Conv2d(2*out_channel, out_channel, 3, 1, 1)
        self.df3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3),nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2))
        self.df5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.2))
        self.conv1_3 = nn.Sequential(nn.Conv2d(3*out_channel,out_channel,1,1),SEBlock(out_channel))
    def forward(self, x):
        out_df1=self.df1(x)
        conv1_1=self.conv1_1(torch.cat([x,out_df1],dim=1))
        conv1_2=self.conv1_2(torch.cat([x,out_df1],dim=1))
        df3 = self.df3(conv1_1)
        df5 = self.df5(conv1_2)
        compact=self.conv1_3(torch.cat([out_df1,df3,df5],dim=1))
        return compact
class My_unit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(My_unit, self).__init__()
        if settings.dilation is True:
            self.df1 = nn.Sequential(nn.Conv2d(in_channel, out_channel,3,1,1,1),nn.LeakyReLU(0.2))
            self.conv1_1 = nn.Conv2d(2*out_channel, out_channel, 1, 1)
            self.conv1_2 = nn.Conv2d(2*out_channel, out_channel, 3, 1, 1)
            self.df3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3), nn.LeakyReLU(0.2))
            self.df5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), nn.LeakyReLU(0.2))
            self.conv1_3 = nn.Sequential(nn.Conv2d(3*out_channel,out_channel,1,1),SEBlock(out_channel))
        else:
            self.df1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2))
            self.conv1_1 = nn.Conv2d(2 * out_channel, out_channel, 1, 1)
            self.conv1_2 = nn.Conv2d(2 * out_channel, out_channel, 3, 1, 1)
            self.df3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2))
            self.df5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2))
            self.conv1_3 = nn.Sequential(nn.Conv2d(3 * out_channel, out_channel, 1, 1), SEBlock(out_channel))
    def forward(self, x):
        out_df1 = self.df1(x)
        conv1_1 = self.conv1_1(torch.cat([x, out_df1], dim=1))
        conv1_2 = self.conv1_2(torch.cat([x, out_df1], dim=1))
        df3 = self.df3(conv1_1)
        df5 = self.df5(conv1_2)
        compact = self.conv1_3(torch.cat([out_df1, df3, df5], dim=1))
        return compact
Unit = {
    'traditional': Tradition,
    'res': Res,
    'dilation': Dilation,
    'GRU': ConvGRU,
    'my_unit':My_unit
}[settings.unit]
class My_blocks(nn.Module):
    def __init__(self,in_channel):
        super(My_blocks, self).__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel
        self.num = settings.res_block_num
        self.res = nn.ModuleList()
        self.cat_1 = nn.ModuleList()
        self.cat_2 = nn.ModuleList()
        self.cat_dense=nn.ModuleList()
        for _ in range(self.num):
            self.res.append(Unit(self.in_channel, self.out_channel))
        if settings.connection_style == 'dense_connection':
            for i in range(self.num+1):
                self.cat_dense.append(nn.Conv2d((i+1)*self.out_channel,self.out_channel,1,1))
        if settings.connection_style == 'multi_short_skip_connection':
            for _ in range(int(self.num/2)-1):
                self.cat_1.append(nn.Conv2d(2*self.in_channel,self.out_channel,1,1))
            for i in range(int(self.num/2)):
                self.cat_2.append(nn.Conv2d((i+2)*self.in_channel,self.out_channel,1,1))
        elif settings.connection_style == 'symmetric_connection':
            for _ in range(int(self.num/2)):
                self.cat_2.append(nn.Conv2d(2*self.in_channel,self.out_channel,1,1))
    def forward(self, x):
        if settings.connection_style == 'dense_connection':
            out=[]
            out.append(x)
            for i in range(self.num):
                x=self.res[i](x)
                out.append(x)
                mid = []
                #print(out[-1].size())
                for j in range(i+2):
                    mid.append(out[j])
                x=self.cat_dense[i+1](torch.cat(mid, dim=1))
            return x
        if settings.connection_style == 'multi_short_skip_connection':
            out=[]
            out.append(x)
            for i in range(self.num):
                x=self.res[i](x)
                out.append(x)
                if i%2==0 & i>=2:
                    odd=[] #odd：奇数
                    for j in range(i):
                        odd.append(out[2*j+1])
                    x=self.cat_1[int((i-2)/2)](torch.cat(odd,dim=1))
                if i%2==1:
                    even=[]
                    even.append(out[0])
                    even.append(out[2])
                    if i>=3:
                        for s in range(int((i-1)/2)):
                            even.append(out[2*(s+2)])
                    x=self.cat_2[int((i-1)/2)](torch.cat(even,dim=1))
            return x
        elif settings.connection_style == 'symmetric_connection':
            out=[]
            out.append(x)
            for i in range(self.num):
                x=self.res[i](x)
                out.append(x)
                if i >= (int(self.num/2)):
                    x=self.cat_2[int(i-int(self.num/2))](torch.cat([out[-1],out[(-2)*(i-int(self.num/2)+1)-1]],dim=1))
            return x
        elif settings.connection_style == 'no_connection':
            for i in range(self.num):
                x=self.res[i](x)
            return x
class RESCAN(nn.Module):
    def __init__(self):
        super(RESCAN, self).__init__()
        if settings.unit=='dilation':
            channel_num=13
        elif settings.unit=='res':
            channel_num = 15
        elif settings.unit=='traditional':
            channel_num = 15
        elif settings.unit=='GRU':
            channel_num = 13
        elif settings.unit=='my_unit':
            channel_num = settings.feature_map_num
        self.extract = nn.Sequential(nn.Conv2d(3,channel_num,3,1,1),nn.LeakyReLU(0.2),SE(channel_num))
        self.dense = My_blocks(channel_num)
        self.exit = nn.Sequential(
            nn.Conv2d(channel_num,channel_num,3,1,1),
            nn.LeakyReLU(0.2),
            SE(channel_num),
            nn.Conv2d(channel_num, 3, 1, 1)
        )
    def forward(self, x):
        extract=self.extract(x)
        res = self.dense(extract)
        final_out = self.exit(res)
        out=[]
        out.append(x-final_out)
        feature=[]
        feature.append(res)
        return out,feature
class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(3,6,8,11), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features

#ts = torch.Tensor(16, 3, 64, 64).cuda()
#vr = Variable(ts)
#net = RESCAN().cuda()
#print(net)
#oups = net(vr)
#for oup in oups:
#    print(oup.size())
