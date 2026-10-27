import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from datasets.registry import build_dataset
from models.spectral_stem import SpectralStem
from metrics.segmentation import confusion_metrics

class CoreTests(unittest.TestCase):
    def test_spectral_stem_preserves_normalized_rgb(self):
        stem = SpectralStem(['rgb_r','rgb_g','rgb_b','ms_g','ms_r','ms_re','ms_nir'])
        x = torch.randn(2,7,8,8)
        torch.testing.assert_close(stem(x), x[:,:3], rtol=0, atol=0)
        stem(x).sum().backward()
        self.assertIsNotNone(stem.projection.weight.grad)

    def test_metrics_known_confusion(self):
        m=confusion_metrics(1,1,1,1)
        for k in ('foreground_iou','background_iou','miou'): self.assertAlmostEqual(m[k],1/3)
        for k in ('dice','precision','recall'): self.assertAlmostEqual(m[k],.5)

    def test_weedyrice_samples_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for d in ('RGB','Masks','Multispectral'): (p/d).mkdir()
            (p/'train_list.txt').write_text('a.JPG\n')
            Image.fromarray(np.full((4,5,3),128,np.uint8)).save(p/'RGB'/'a.JPG')
            Image.fromarray(np.array([[0,255,0,255,0]]*4,np.uint8)).save(p/'Masks'/'a.png')
            config={'root':str(p),'input_mode':'rgb','size':8}
            sample=build_dataset('weedyrice','train',config)
            self.assertEqual(tuple(sample[0]['image'].shape),(3,8,8))
            self.assertEqual(set(torch.unique(sample[0]['mask']).tolist()),{0,1})
            (p/'Masks'/'a.png').unlink()
            with self.assertRaises(FileNotFoundError): build_dataset('weedyrice','train',config)

    def test_invalid_mask_value_is_rejected_before_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for d in ('RGB','Masks','Multispectral'): (p/d).mkdir()
            (p/'train_list.txt').write_text('a.JPG\n')
            Image.fromarray(np.zeros((10,10,3),np.uint8)).save(p/'RGB'/'a.JPG')
            mask=np.zeros((10,10),np.uint8); mask[0,0]=42
            Image.fromarray(mask).save(p/'Masks'/'a.png')
            dataset=build_dataset('weedyrice','train',{'root':str(p),'input_mode':'rgb','size':2})
            with self.assertRaises(ValueError): dataset[0]
