import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self,input_dim, output_dim, encoder):
        super(Protonet, self).__init__()
        # 简单的两层全连接网络来实现原型的变换
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

        self.encoder = encoder
    # 原型内聚损失
    def prototype_cohesion_loss(embeddings, transformed_prototypes, support_set):
        loss = 0.0
        for i, prototype in enumerate(transformed_prototypes):
            # 获取类别 i 的样本嵌入
            class_samples = embeddings[support_set == i]

            # 计算类别 i 中所有样本与变换原型之间的距离
            loss += torch.sum((class_samples - prototype) ** 2)

        return loss / embeddings.size(0)
    # 原型分离损失
    def prototype_separation_loss(transformed_prototypes, delta=1.0):
        num_classes = transformed_prototypes.size(0)
        loss = 0.0

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distance = torch.norm(transformed_prototypes[i] - transformed_prototypes[j], p=2)
                # 强制要求类别 i 和 j 之间的距离大于 delta
                loss += torch.max(torch.tensor(0.0), delta - distance)

        return loss
    # def loss(self, sample):
    #     xs = Variable(sample['xs']) # support
    #     xq = Variable(sample['xq']) # query
    #
    #     n_class = xs.size(0)
    #     assert xq.size(0) == n_class
    #     n_support = xs.size(1)
    #     n_query = xq.size(1)
    #
    #     target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    #     target_inds = Variable(target_inds, requires_grad=False)
    #
    #     if xq.is_cuda:
    #         target_inds = target_inds.cuda()
    #
    #     x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
    #                    xq.view(n_class * n_query, *xq.size()[2:])], 0)
    #
    #     z = self.encoder.forward(x)
    #     z_dim = z.size(-1)
    #
    #     z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
    #     zq = z[n_class*n_support:]
    #
    #     dists = euclidean_dist(zq, z_proto)
    #
    #     log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
    #
    #     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    #
    #     _, y_hat = log_p_y.max(2)
    #     acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    #
    #     return loss_val, {
    #         'loss': loss_val.item(),
    #         'acc': acc_val.item()
    #     }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)
