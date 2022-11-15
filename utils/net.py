from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
import os

class BaseFeatureExtractor(nn.Module):
    '''
    From https://github.com/thuml/Universal-Domain-Adaptation
    a base class for feature extractor
    '''
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    """
    modefied from https://github.com/thuml/Universal-Domain-Adaptation
    implement ResNet50 as backbone, but the last fc layer is removed
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                model_resnet = models.resnet50(pretrained=False)
                model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        mod = list(model_resnet.children())
        mod.pop()
        self.feature_extractor = nn.Sequential(*mod)
        self.output_dim = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x


class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim, hidden_mlp=2048, feat_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
                            nn.Linear(in_dim, hidden_mlp),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_mlp, feat_dim))
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls


