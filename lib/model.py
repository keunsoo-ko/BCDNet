import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layer(block, num_of_layer, c):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block(c, c))
    return nn.Sequential(*layers)


class Sim_Block(nn.Module):
    def __init__(self, c, topk=7):
        super(Sim_Block, self).__init__()
        self.topk = topk
        self.mix = nn.Conv3d(c, c, (topk, 1, 1), padding=(0, 0, 0))
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.dividing(x)
        return self.relu(self.mix(x).squeeze(2))

    def dividing(self, x):
        cut = 1 # if increasing the value of cut, save the memory of GPU
        b, c, h, w = x.shape
        step = h*w // cut
        X_ = x.reshape(b, c, -1)
        X = X_ / (X_**2).sum(1, keepdim=True)**0.5
        topks = []
        indices = []
        for i in range(cut):
            score = self.SS(X[..., i*step:(i+1)*step], X)
            topk, indice = torch.topk(score, self.topk, largest=True)
            topks.append(topk.permute(0, 2, 1))
            indices.append(indice.permute(0, 2, 1))
        topks = torch.cat(topks, 2)[:, None].repeat(1, c, 1, 1)
        indices = torch.cat(indices, 2)[:, None].repeat(1, c, 1, 1)
        high_d = torch.gather(X_[:, :, None].repeat(1, 1, self.topk, 1), 3, indices) * topks
        return high_d.reshape(b, c, self.topk, h, w)

    def SS(self, x1, x2):
        Xt = x1.permute(0, 2, 1)
        score = torch.bmm(Xt, x2)
        return score

    def SIM(self, x):
        ntop = self.topk
        b, c, h, w = x.shape
        X_ = x.reshape(b, c, -1)
        X = X_ / (X_**2).sum(1, keepdim=True)**0.5
        Xt = X.permute(0, 2, 1)
        score = torch.bmm(Xt, X)
        topk, indices = torch.topk(score, ntop, largest=True)
        indices = indices.permute(0, 2, 1)[:, None].repeat(1, c, 1, 1)
        topk = topk.permute(0, 2, 1)[:, None].repeat(1, c, 1, 1)
        X_ = X_[:, :, None]
        high_d = torch.gather(X_.repeat(1, 1, ntop, 1), 3, indices)
        high_d *= topk
        return high_d.reshape(b, c, ntop, h, w)


class Conv_ReLU_Block(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel, stride=stride, padding=kernel//2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Res_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv = nn.Conv2d(c_out*3, c_out, kernel_size=1, stride=1)
        self.W = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        res = x
        x = self.relu(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], 1))
        return self.conv(x) + res*self.W

class Multi_Block(nn.Module):
    def __init__(self, c_in, c_out, k_size = 3):
        super(Multi_Block, self).__init__()
        self.k_size = k_size
        self.c_out = c_out
        self.c_in = c_in
        self.M_out = nn.Conv2d(in_channels=c_in, out_channels=c_out*self.k_size**2, kernel_size=3, stride=1, padding=1, bias=True)
        self.CF = nn.Conv2d(c_in, c_in*c_out*(self.k_size**2), 1, stride=1)


    def forward(self, feature, filters, cf):
        b, c, h, w = feature.shape
        M = self.M_out(filters).\
            reshape(b, self.k_size**2, 1, self.c_out, h, w) # b, k^2, 1, c_out, h, w
        K = self.CF(cf).reshape(b, self.k_size**2, self.c_in, self.c_out, 1, 1)
        df = M*K
        x_ = F.pad(feature, (self.k_size // 2, self.k_size // 2,
                       self.k_size // 2, self.k_size // 2))
        x_ = x_.unfold(2, self.k_size, 1).unfold(3, self.k_size, 1)
        x_ = x_.permute(0, 4, 5, 1, 2, 3).reshape(b, self.k_size**2, -1, 1, h, w)

        return (x_ * df).sum([1, 2])    


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        layers = []
        for i in range(4):
           layers.append(Conv_ReLU_Block(32, 32))
           layers.append(Conv_ReLU_Block(32, 32, stride=2))

        self.fc1 = Conv_ReLU_Block(32, 32, kernel=1)
        self.fc2 = nn.Conv2d(32, 1, 1, stride=1)
        self.estimate = nn.Sequential(*layers)

    def forward(self, x):
        x = self.estimate(x)
        x = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        filters = self.fc1(x)
        n_level = self.fc2(filters)
        return filters, n_level.squeeze(-1).squeeze(-1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = make_layer(Res_Block, 3, 32)
        self.main_ = make_layer(Res_Block, 1, 32)
        self.get_feature = make_layer(Res_Block, 2, 32)
        self.get_filters = make_layer(Conv_ReLU_Block, 2, 32)
        self.input = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.Rec = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.sim = Sim_Block(32)
        self.output2 = Multi_Block(32, 3)

        self.n_estimate = NE()

    def reduction(self, x, feature, CF):
        residual = x[:, :3]

        x = self.input(x)
        out = self.relu(x)

        if feature is not None:
            out = torch.cat([out, feature], 1)
            out = F.relu(self.Rec(out))

        x_ = self.main(out)
        x_ = self.main_(x_)
        x_ = self.sim(x_)

        feature = self.get_feature(x_)

        out = self.get_filters(feature)

        return self.output2(feature, out, CF) + residual, x_

    def forward(self, real, syn, f_real, f_syn, CF_real, CF_syn):
        n_level = None
        if CF_real is None:
            CF_real, n_level = self.n_estimate(real[:, :3], syn[:, :3])
            CF_syn, _ = self.n_estimate(syn[:, :3], real[:, :3])

        real, f_real = self.reduction(real, f_real, CF_real) 
        syn, f_syn = self.reduction(syn, f_syn, CF_syn)

        return real, syn, f_real, f_syn, n_level, CF_real, CF_syn


class NE(nn.Module):
    def __init__(self):
        super(NE, self).__init__()
        self.input = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.n_estimate = Estimator()
        self.fc1 = Conv_ReLU_Block(64, 64, kernel=1)
        self.fc2 = nn.Conv2d(64, 2, 1, stride=1)

        self.f_fc1 = Conv_ReLU_Block(32, 32, kernel=1)
        self.f_fc2 = Conv_ReLU_Block(32, 32, kernel=1)

    def forward(self, real, syn):
        real = self.n_estimate(F.relu(self.input(real)))
        syn = self.n_estimate(F.relu(self.input(syn)))
        x = torch.cat([real, syn], 1)
        n_level = self.fc2(F.relu(self.fc1(x)))
        CF = F.relu(self.f_fc2(F.relu(self.f_fc1(real))))
        return CF, n_level.squeeze(-1).squeeze(-1)

class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        layers = []
        for i in range(4):
           layers.append(Conv_ReLU_Block(32, 32))
           layers.append(Conv_ReLU_Block(32, 32, stride=2))
        self.estimate = nn.Sequential(*layers)

    def forward(self, x):
        x = self.estimate(x)
        x = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        return x
