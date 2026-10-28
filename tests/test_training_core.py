import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from engine.losses import structure_loss
from engine.checkpoint import save_checkpoint, load_checkpoint
from models.factory import SegmentationModel

class TrainingTests(unittest.TestCase):
 def test_loss_backward_and_checkpoint_restore(self):
  torch.manual_seed(2)
  model=SegmentationModel(nn.Conv2d(3,1,1),['rgb_r','rgb_g','rgb_b','ms_nir']); opt=torch.optim.AdamW(model.parameters(),lr=.01)
  x=torch.randn(2,4,10,10); target=(torch.rand(2,1,10,10)>.5).float()
  before=model.stem.projection.weight.detach().clone(); loss=structure_loss(model(x)['logits'],target); loss.backward();opt.step()
  self.assertGreater((model.stem.projection.weight[:,3:]-before[:,3:]).abs().sum().item(),0)
  model.eval(); expected=model(x)['logits'].detach().clone()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'checkpoint.pt'; save_checkpoint(p,model,opt,None,{'name':'toy'},1,.2)
   with torch.no_grad(): model.stem.projection.weight.zero_()
   state=load_checkpoint(p,model,opt); self.assertEqual(state['epoch'],1)
   torch.testing.assert_close(model(x)['logits'],expected)
