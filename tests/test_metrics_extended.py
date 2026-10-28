import unittest
import math
import torch
from metrics.segmentation import confusion_metrics, confusion_from_logits, aggregate_metrics

class MetricsTests(unittest.TestCase):
    def test_empty_foreground_is_undefined_not_perfect(self):
        m=confusion_metrics(0,0,0,8)
        self.assertTrue(math.isnan(m['foreground_iou']))
        self.assertTrue(math.isnan(m['dice']))
        self.assertEqual(m['background_iou'],1)
        self.assertEqual(m['miou'],1)
    def test_false_positive_and_empty_prediction(self):
        self.assertEqual(confusion_metrics(0,1,0,7)['foreground_iou'],0)
        self.assertEqual(confusion_metrics(0,0,1,7)['recall'],0)
        self.assertTrue(math.isnan(confusion_metrics(0,0,1,7)['precision']))
    def test_micro_macro_distinct(self):
        a=confusion_from_logits(torch.tensor([[[[10.,-10.]]]]),torch.tensor([[[[1.,0.]]]]))
        b=confusion_from_logits(torch.tensor([[[[-10.,-10.]]]]),torch.tensor([[[[1.,1.]]]]))
        result=aggregate_metrics([a,b]); self.assertAlmostEqual(result['micro']['foreground_iou'],1/3); self.assertAlmostEqual(result['macro']['foreground_iou'],.5)
