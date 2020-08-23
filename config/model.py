import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations,product
import math

class Inner_scale_connection_block(nn.Module):
    def __init__(self):
        super(Inner_scale_connection_block, self).__init__()
        self.channel = settings.channel
        self.scale_num = settings.scale_num
        self.conv_num = settings.conv_num
        self.scale1 = nn.ModuleList()
        self.scale2 = nn.ModuleList()
        self.scale4 = nn.ModuleList()
        self.scale8 = nn.ModuleList()
        if settings.scale_num == 4:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale8.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.fusion84 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.pooling8 = nn.MaxPool2d(8, 8)
            self.pooling4 = nn.MaxPool2d(4, 4)
            self.pooling2 = nn.MaxPool2d(2, 2)
            self.fusion_all = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        elif settings.scale_num == 3:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.pooling4 = nn.MaxPool2d(4, 4)
            self.pooling2 = nn.MaxPool2d(2, 2)
            self.fusion_all = nn.Sequential(nn.Conv2d(3 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        elif settings.scale_num == 2:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
                self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
            self.pooling2 = nn.MaxPool2d(2, 2)
            self.fusion_all = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        elif settings.scale_num == 1:
            for i in range(self.conv_num):
                self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        if settings.scale_num == 4:
            feature8 = self.pooling8(x)
            b8, c8, h8, w8 = feature8.size()
            feature4 = self.pooling4(x)
            b4, c4, h4, w4 = feature4.size()
            feature2 = self.pooling2(x)
            b2, c2, h2, w2 = feature2.size()
            feature1 = x
            b1, c1, h1, w1 = feature1.size()
            for i in range(self.conv_num):
                feature8 = self.scale8[i](feature8)
            scale8 = feature8
            feature4 = self.fusion84(torch.cat([feature4, F.upsample(scale8, [h4, w4])], dim=1))
            for i in range(self.conv_num):
                feature4 = self.scale4[i](feature4)
            scale4 = feature4
            feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))
            for i in range(self.conv_num):
                feature2 = self.scale2[i](feature2)

            scale2 = feature2
            feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
            for i in range(self.conv_num):
                feature1 = self.scale1[i](feature1)
            scale1 = feature1
            fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1]), F.upsample(scale8, [h1, w1])], dim=1))
            return fusion_all + x
        elif settings.scale_num == 3:
            feature4 = self.pooling4(x)
            b4, c4, h4, w4 = feature4.size()
            feature2 = self.pooling2(x)
            b2, c2, h2, w2 = feature2.size()
            feature1 = x
            b1, c1, h1, w1 = feature1.size()

            for i in range(self.conv_num):
                feature4 = self.scale4[i](feature4)
            scale4 = feature4
            feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))
            for i in range(self.conv_num):
                feature2 = self.scale2[i](feature2)
            scale2 = feature2
            feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
            for i in range(self.conv_num):
                feature1 = self.scale1[i](feature1)
            scale1 = feature1
            fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1])],dim=1))
            return fusion_all + x
        elif settings.scale_num == 2:
            feature2 = self.pooling2(x)
            b2, c2, h2, w2 = feature2.size()
            feature1 = x
            b1, c1, h1, w1 = feature1.size()

            for i in range(self.conv_num):
                feature2 = self.scale2[i](feature2)
            scale2 = feature2
            feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
            for i in range(self.conv_num):
                feature1 = self.scale1[i](feature1)
            scale1 = feature1
            fusion_all = self.fusion_all(
                torch.cat([scale1, F.upsample(scale2, [h1, w1])], dim=1))
            return fusion_all + x
        elif settings.scale_num == 1:
            feature1 = x
            b1, c1, h1, w1 = feature1.size()
            scale1 = self.scale1(feature1)
            fusion_all = scale1
            return fusion_all + x
            
class Cross_scale_fusion_block(nn.Module):
    def __init__(self):
        super(Cross_scale_fusion_block, self).__init__()
        self.channel_num = settings.channel
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion3_1 = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion3_2 = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion3_3 = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_3 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.pooling4 = nn.MaxPool2d(4, 4)

    def forward(self, x):
        input = x
        encoder1 = self.encoder_conv1(x)
        b1, c1, h1, w1 = encoder1.size()
        pooling1 = self.pooling2(encoder1)
        encoder2 = self.encoder_conv2(pooling1)
        b2, c2, h2, w2 = encoder2.size()
        pooling2 = self.pooling2(encoder2)
        encoder3 = self.encoder_conv3(pooling2)
        b3, c3, h3, w3 = encoder3.size()
        encoder1_resize1 = F.upsample(encoder1, [h3, w3])
        encoder2_resize1 = F.upsample(encoder2, [h3, w3])
        encoder3_resize1 = F.upsample(encoder3, [h3, w3])
        fusion3_1 = self.fusion3_1(torch.cat([encoder1_resize1, encoder2_resize1, encoder3_resize1], dim=1))
        encoder1_resize2 = F.upsample(encoder1, [h2, w2])
        encoder2_resize2 = F.upsample(encoder2, [h2, w2])
        encoder3_resize2 = F.upsample(encoder3, [h2, w2])
        fusion3_2 = self.fusion3_2(torch.cat([encoder1_resize2, encoder2_resize2, encoder3_resize2], dim=1))
        encoder1_resize3 = F.upsample(encoder1, [h1, w1])
        encoder2_resize3 = F.upsample(encoder2, [h1, w1])
        encoder3_resize3 = F.upsample(encoder3, [h1, w1])
        fusion3_3 = self.fusion3_3(torch.cat([encoder1_resize3, encoder2_resize3, encoder3_resize3], dim=1))

        decoder_conv1 = self.decoder_conv1(self.fusion2_1(torch.cat([fusion3_1, F.upsample(encoder3, [h3, w3])], dim=1)))
        decoder_conv2 = self.decoder_conv2(self.fusion2_2(torch.cat([fusion3_2, F.upsample(decoder_conv1, [h2, w2])], dim=1)))
        decoder_conv3 = self.decoder_conv3(self.fusion2_3(torch.cat([fusion3_3, F.upsample(decoder_conv2, [h1, w1])], dim=1)))
        return decoder_conv3 + input


class Encoder_decoder_block(nn.Module):
    def __init__(self):
        super(Encoder_decoder_block, self).__init__()
        self.channel_num = settings.channel
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_3 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.pooling2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        input = x
        encoder1 = self.encoder_conv1(x)
        b1, c1, h1, w1 = encoder1.size()
        pooling1 = self.pooling2(encoder1)
        encoder2 = self.encoder_conv2(pooling1)
        b2, c2, h2, w2 = encoder2.size()
        pooling2 = self.pooling2(encoder2)
        encoder3 = self.encoder_conv3(pooling2)

        decoder_conv1 = self.decoder_conv1(encoder3)
        decoder_conv2 = self.decoder_conv2(self.fusion2_2(torch.cat([F.upsample(encoder2, [h2, w2]), F.upsample(decoder_conv1, [h2, w2])], dim=1)))
        decoder_conv3 = self.decoder_conv3(self.fusion2_3(torch.cat([F.upsample(encoder1, [h1, w1]), F.upsample(decoder_conv2, [h1, w1])], dim=1)))
        return decoder_conv3 + input


Scale_block = Inner_scale_connection_block


class ConvDirec(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        x = self.conv(x)
        x = self.relu(x)
        return x, None


class ConvRNN(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_x = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_h = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            h = F.tanh(self.conv_x(x))
        else:
            h = F.tanh(self.conv_x(x) + self.conv_h(h))

        h = self.relu(h)
        return h, h


class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(h)
        return h, h


class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, pair=None):
        if pair is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            h, c = pair
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        h = self.relu(h)
        return h, [h, c]


RecUnit = {
    'Conv': ConvDirec,
    'RNN': ConvRNN,
    'GRU': ConvGRU,
    'LSTM': ConvLSTM,
}[settings.uint]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.unit_num = settings.Num_encoder
        self.units = nn.ModuleList()
        self.channel_num = settings.channel
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Scale_block())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel_num, self.channel_num, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        catcompact = []
        catcompact.append(x)
        feature = []
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            feature.append(tmp)
            catcompact.append(tmp)
            out = self.conv1x1[i](torch.cat(catcompact, dim=1))
        return out, feature


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.unit_num = settings.Num_encoder
        self.units = nn.ModuleList()
        self.channel_num = settings.channel
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Scale_block())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel_num, self.channel_num, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x, feature):
        catcompact=[]
        catcompact.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out + feature[i])
            catcompact.append(tmp)
            out = self.conv1x1[i](torch.cat(catcompact, dim=1))
        return out


class DenseConnection(nn.Module):
    def __init__(self, unit, unit_num):
        super(DenseConnection, self).__init__()
        self.unit_num = unit_num
        self.channel = settings.channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(unit())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        cat = []
        cat.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            cat.append(tmp)
            out = self.conv1x1[i](torch.cat(cat, dim=1))
        return out


class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        self.channel = 24
        self.unit_num = 24
        self.enterBlock = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.derain_net = DenseConnection(Cross_scale_fusion_block, self.unit_num)
        self.exitBlock = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        image_feature = self.enterBlock(x)
        rain_feature = self.derain_net(image_feature)
        rain = self.exitBlock(rain_feature)
        derain = x - rain
        return derain

class Multi_model_fusion_learning(nn.Module):
    def __init__(self):
        super(Multi_model_fusion_learning, self).__init__()
        self.channel_num = settings.channel
        self.convert = nn.Sequential(
            nn.Conv2d(3, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_scale1 = Encoder()
        self.encoder_scale2 = Encoder()
        self.encoder_scale4 = Encoder()
        if settings.Net_cross is True:
            self.fusion_1_2_1 = nn.Sequential(
                nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
                nn.LeakyReLU(0.2))
            self.fusion_1_2_2 = nn.Sequential(
                nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
                nn.LeakyReLU(0.2))
            self.fusion_2_2_1 = nn.Sequential(
                nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
                nn.LeakyReLU(0.2))
            self.fusion_2_2_2 = nn.Sequential(
                nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
                nn.LeakyReLU(0.2))
        self.decoder_scale1 = Decoder()
        self.decoder_scale2 = Decoder()
        self.decoder_scale4 = Decoder()
        self.rec1_1 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec1_2 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec1_4 = RecUnit(self.channel_num, self.channel_num, 3, 1)

        self.rec2_1 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec2_2 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec2_4 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.merge = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 3, 3, 1, 1)
        )
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.pooling4 = nn.MaxPool2d(4, 4)

    def forward(self, x):
        convert = self.convert(x)
        feature1 = convert
        feature2 = self.pooling2(convert)
        feature4 = self.pooling4(convert)
        b1, c1, h1, w1 = feature1.size()
        b2, c2, h2, w2 = feature2.size()
        b4, c4, h4, w4 = feature4.size()
        scale1_encoder, scale1_feature = self.encoder_scale1(feature1)
        scale2_encoder, scale2_feature = self.encoder_scale2(feature2)
        scale4_encoder, scale4_feature = self.encoder_scale4(feature4)

        if settings.Net_cross is True:
            current1_4, rec1_4 = self.rec1_4(scale4_encoder)
            rec1_4_ori = rec1_4
            if settings.uint == "LSTM":
                rec1_4[0], rec1_4[1] = F.upsample(rec1_4_ori[0], [h2,w2]), F.upsample(rec1_4_ori[1], [h2,w2])
                current1_2, rec1_2 = self.rec1_2(scale2_encoder, rec1_4)
                rec1_2_ori = rec1_2
                rec1_2[0], rec1_2[1] = F.upsample(rec1_2_ori[0], [h1, w1]), F.upsample(rec1_2_ori[1], [h1, w1])
                rec1_4[0], rec1_4[1] = F.upsample(rec1_4_ori[0], [h1, w1]), F.upsample(rec1_4_ori[1], [h1, w1])
                current1_1, rec1_1 = self.rec1_1(scale1_encoder, [self.fusion_1_2_1(torch.cat([rec1_2[0], rec1_4[0]],dim=1)), self.fusion_1_2_2(torch.cat([rec1_2[1], rec1_4[1]], dim=1))])
            else:
                rec1_4 = F.upsample(rec1_4_ori, [h2, w2])
                current1_2, rec1_2 = self.rec1_2(scale2_encoder, rec1_4)
                rec1_2_ori = rec1_2
                rec1_2 = F.upsample(rec1_2_ori, [h1, w1])
                rec1_4 = F.upsample(rec1_4_ori, [h1, w1])
                current1_1, rec1_1 = self.rec1_1(scale1_encoder, self.fusion_1_2_1(torch.cat([rec1_2, rec1_4], dim=1)))
        else:
            current1_1 = scale1_encoder
            current1_2 = scale2_encoder
            current1_4 = scale4_encoder

        if settings.Net_cross is True:
            current2_1, rec2_1 = self.rec2_1(current1_1)
            rec2_1_ori = rec2_1
            if settings.uint == "LSTM":
                rec2_1[0], rec2_1[1] = F.upsample(rec2_1_ori[0], [h2, w2]), F.upsample(rec2_1_ori[1], [h2, w2])
                current2_2, rec2_2 = self.rec2_2(current1_2, rec2_1)
                rec2_2_ori = rec2_2
                rec2_2[0], rec2_2[1] = F.upsample(rec2_2_ori[0], [h4, w4]), F.upsample(rec2_2_ori[1], [h4, w4])
                rec2_1[0], rec2_1[1] = F.upsample(rec2_1_ori[0], [h4, w4]), F.upsample(rec2_1_ori[1], [h4, w4])
                current2_4, rec2_4 = self.rec2_4(current1_4, [self.fusion_2_2_1(torch.cat([rec2_1[0],rec2_2[0]],dim=1)), self.fusion_2_2_2(torch.cat([rec2_1[1], rec2_2[1]],dim=1))])
            else:
                rec2_1 = F.upsample(rec2_1_ori, [h2, w2])
                current2_2, rec2_2 = self.rec2_2(current1_2, rec2_1)
                rec2_2_ori = rec2_2
                rec2_2 = F.upsample(rec2_2_ori, [h4, w4])
                rec2_1 = F.upsample(rec2_1_ori, [h4, w4])
                current2_4, rec2_4 = self.rec2_4(current1_4, self.fusion_2_2_1(torch.cat([rec2_1, rec2_2],dim=1)))
        else:
            current2_1 = current1_1
            current2_2 = current1_2
            current2_4 = current1_4
        scale1_decoder = self.decoder_scale1(current2_1, scale1_feature)
        scale2_decoder = self.decoder_scale2(current2_2, scale2_feature)
        scale4_decoder = self.decoder_scale4(current2_4, scale4_feature)
        merge = self.merge(torch.cat([scale1_decoder, F.upsample(scale2_decoder,[h1,w1]), F.upsample(scale4_decoder,[h1,w1])],dim=1))

        return x-merge


class Single_model_Learning(nn.Module):
    def __init__(self):
        super(Single_model_Learning, self).__init__()
        self.channel_num = settings.channel
        self.convert = nn.Sequential(
            nn.Conv2d(3, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.merge = nn.Sequential(
            nn.Conv2d(self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 3, 3, 1, 1)
        )

    def forward(self, x):
        convert = self.convert(x)
        encoder, feature = self.encoder(convert)
        decoder = self.decoder(encoder, feature)

        merge = self.merge(decoder)

        return x-merge


Net = Single_model_Learning if settings.single else Multi_model_fusion_learning
