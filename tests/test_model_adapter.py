import unittest
import torch
from torch import nn
from models.factory import SegmentationModel, parameter_summary, verify_sam3_weights

class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Conv2d(3,4,1); self.head=nn.Conv2d(4,1,1)
    def forward(self,x):
        y=self.head(self.encoder(x)); return y,y+1,y+2

class ModelTests(unittest.TestCase):
    def test_output_and_stem_train_update(self):
        x=torch.randn(2,7,8,8); m=SegmentationModel(Toy(),['rgb_r','rgb_g','rgb_b','ms_g','ms_r','ms_re','ms_nir'])
        opt=torch.optim.SGD(m.parameters(),lr=.1); before=m.stem.projection.weight.detach().clone()
        output=m(x); self.assertEqual(len(output['aux']),2); output['logits'].mean().backward(); opt.step()
        self.assertTrue(torch.isfinite(m.stem.projection.weight.grad).all()); self.assertGreater(float((before[:,3:]-m.stem.projection.weight.detach()[:,3:]).abs().sum()),0)
        self.assertGreater(parameter_summary(m)['stem']['trainable'],0)
    def test_rgb_is_identity(self):
        m=SegmentationModel(Toy(),['rgb_r','rgb_g','rgb_b']); self.assertIsInstance(m.stem,nn.Identity)
    def test_sam3_strict_failure(self):
        with self.assertRaises(ValueError): verify_sam3_weights({'layer.weight':torch.zeros(2,2)}, {'layer.weight':torch.zeros(3,2)})
        with self.assertRaises(ValueError): verify_sam3_weights({'layer.weight':torch.zeros(2,2)}, {})

    def test_missing_or_wrong_pretrained_is_explicit(self):
        from models.factory import build_model
        with self.assertRaises(FileNotFoundError):
            build_model('sam2_unet',['rgb_r','rgb_g','rgb_b'],{'pretrained':'/no/such/checkpoint.pt','pretrained_version':'sam2'})
        with self.assertRaises(ValueError):
            build_model('unknown',['rgb_r','rgb_g','rgb_b'],{})
        with self.assertRaises(ValueError):
            build_model('sam2_unet',['rgb_r','rgb_g','rgb_b'],{'num_classes':3})
