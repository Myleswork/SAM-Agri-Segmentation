import unittest
import torch
from torch import nn
from engine.evaluator import evaluate

class EvaluatorTests(unittest.TestCase):
 def test_aggregation_and_coverage_groups(self):
  class Identity(nn.Module):
   def forward(self,x): return {'logits':x,'aux':[]}
  batch={'image':torch.tensor([[[[10.,-10.]]],[[[-10.,-10.]]]]),'mask':torch.tensor([[[[1.,0.]]],[[[1.,1.]]]]),'sample_id':['a','b']}
  out,rows=evaluate(Identity(),[batch],'cpu')
  self.assertEqual(len(rows),2); self.assertAlmostEqual(out['micro']['foreground_iou'],1/3)
  self.assertAlmostEqual(out['macro']['foreground_iou'],.5); self.assertEqual(sum(x['count'] for x in out['coverage_groups'].values()),2)
