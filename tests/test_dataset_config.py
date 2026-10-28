import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from datasets.registry import build_dataset, register

class DatasetConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        for x in ('RGB','Masks','Multispectral'): (self.root/x).mkdir()
        (self.root/'train_list.txt').write_text('x.JPG\n')
        Image.fromarray(np.full((4,6,3),128,np.uint8)).save(self.root/'RGB/x.JPG')
        Image.fromarray(np.tile(np.array([0,0,0,255,255,255],dtype=np.uint8),(4,1))).save(self.root/'Masks/x.png')
        for band,value in [('G',20),('R',40),('RE',60),('NIR',80)]:
            Image.fromarray(np.full((4,6,3),value,np.uint8)).save(self.root/f'Multispectral/x_{band}.TIF')
    def tearDown(self): self.tmp.cleanup()
    def base(self): return {'root':str(self.root),'size':[2,3],'input_mode':'rgbms7','ms_scale':100,'rgb_mean':[0,0,0],'rgb_std':[1,1,1], 'channels':{'rgbms7':['rgb_r','rgb_g','rgb_b','ms_nir','ms_re','ms_r','ms_g']}}
    def test_configured_shape_normalization_channel_order(self):
        x=build_dataset('weedyrice','train',self.base())[0]
        self.assertEqual(tuple(x['image'].shape),(7,2,3))
        self.assertAlmostEqual(float(x['image'][0,0,0]),128/255,places=2)
        self.assertAlmostEqual(float(x['image'][3,0,0]),.8,places=5)
        self.assertAlmostEqual(float(x['image'][6,0,0]),.2,places=5)
        self.assertEqual(set(x['mask'].unique().tolist()),{0.,1.})
    def test_invalid_scale(self):
        for value in (0,-1,float('nan'),float('inf')):
            c=self.base(); c['ms_scale']=value
            with self.assertRaises(ValueError): build_dataset('weedyrice','train',c)
    def test_registry_extension(self):
        @register('example_test')
        class Example:
            def __init__(self,split,config): self.split=split
        self.assertEqual(build_dataset('example_test','val',{}).split,'val')
