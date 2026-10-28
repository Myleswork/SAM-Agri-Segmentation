import unittest
import numpy as np
import torch
from PIL import Image
from torch import nn
from engine.inference import predict_arrays, error_overlay

class InferenceTests(unittest.TestCase):
 def test_resize_logits_before_threshold(self):
  class Model(nn.Module):
   def forward(self,x):return {'logits':torch.ones((1,1,3,4)), 'aux':[]}
  p,m=predict_arrays(Model(),torch.zeros(1,3,3,4),'cpu',(10,12))
  self.assertEqual(p.shape,(10,12)); self.assertEqual(m.shape,(10,12));self.assertTrue(m.all())
 def test_error_overlay_colours(self):
  rgb=np.zeros((1,3,3),dtype=np.uint8);truth=np.array([[1,0,1]],dtype=bool);pred=np.array([[1,1,0]],dtype=bool)
  result=error_overlay(rgb,truth,pred)
  self.assertGreater(result[0,0,1],0);self.assertGreater(result[0,1,0],0);self.assertGreater(result[0,2,2],0)

 def test_rgb_preprocess_respects_channel_order(self):
  from engine.inference import preprocess_rgb
  image=Image.fromarray(np.array([[[255,128,0]]],dtype=np.uint8))
  cfg={'size':1,'rgb_mean':[0,0,0],'rgb_std':[1,1,1],'input_mode':'rgb','channels':{'rgb':['rgb_b','rgb_r','rgb_g']}}
  output=preprocess_rgb(image,cfg)
  self.assertEqual(tuple(output.shape),(1,3,1,1))
  self.assertAlmostEqual(float(output[0,0,0,0]),0)
  self.assertAlmostEqual(float(output[0,1,0,0]),1)

 def test_pretrained_config_requires_checkpoint(self):
  from engine.inference import load_pretrained
  with self.assertRaisesRegex(ValueError,'pretrained'):
   load_pretrained({'model':{'pretrained':None}},'cpu')

 def test_binary_mask_uses_adapter_foreground_values(self):
  from engine.inference import binary_mask
  values=np.array([[0,2,255]],dtype=np.uint8)
  self.assertEqual(binary_mask(values,[2]).tolist(),[[False,True,False]])
