import torch
import torch.nn.functional as F

def structure_loss(logits,target):
    if logits.shape[-2:] != target.shape[-2:]: logits=F.interpolate(logits,size=target.shape[-2:],mode='bilinear',align_corners=False)
    weight=1+5*torch.abs(F.avg_pool2d(target,31,stride=1,padding=15)-target)
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    bce=(weight*bce).sum((2,3))/weight.sum((2,3))
    prob=logits.sigmoid(); inter=(prob*target*weight).sum((2,3)); union=((prob+target)*weight).sum((2,3))
    return (bce+1-(inter+1)/(union-inter+1)).mean()
