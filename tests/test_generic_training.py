import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import Dataset
from datasets.registry import register
from engine.trainer import train
from models.factory import SegmentationModel

@register('test_generic_adapter')
class GenericDataset(Dataset):
 def __init__(self,split,config): self.split=split
 def __len__(self): return 2
 def __getitem__(self,index):
  return {'image':torch.ones(3,8,8),'mask':torch.ones(1,8,8),'sample_id':f'{self.split}_{index}'}

class GenericTrainerTest(unittest.TestCase):
 def test_new_dataset_trains_without_dataset_specific_logic(self):
  cfg={'dataset':{'name':'test_generic_adapter','input_mode':'rgb','channels':{'rgb':['rgb_r','rgb_g','rgb_b']}},'model':{'name':'sam2_unet','device':'cpu'},'training':{'epochs':1,'batch_size':1,'lr':.001,'max_train_batches':1,'max_eval_batches':1}}
  toy=SegmentationModel(nn.Conv2d(3,1,1),['rgb_r','rgb_g','rgb_b'])
  with tempfile.TemporaryDirectory() as d, patch('engine.trainer.build_model',return_value=toy):
   result=train(cfg,d)
   self.assertIn('micro',result)
