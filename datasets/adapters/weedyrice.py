from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets.registry import register
from tools.inspect_dataset import indexed_files

MODES={'rgb':('rgb_r','rgb_g','rgb_b'), 'rgbnir4':('rgb_r','rgb_g','rgb_b','ms_nir'), 'rgbms7':('rgb_r','rgb_g','rgb_b','ms_g','ms_r','ms_re','ms_nir')}

@register('weedyrice')
class WeedyRice(Dataset):
    def __init__(self, split, config):
        self.root=Path(config['root']); self.split=split; self.config=config
        self.mode=config.get('input_mode','rgb')
        if self.mode not in MODES: raise ValueError(f'unknown input mode {self.mode}')
        self.channels=MODES[self.mode]; self.size=int(config['size'])
        if self.size <= 0: raise ValueError('size must be positive')
        if len(self.channels)>3 and 'ms_scale' not in config: raise ValueError('explicit ms_scale required; sensor values are not assumed reflectance')
        self.files={d:indexed_files(self.root/d) for d in ('RGB','Masks','Multispectral')}
        self.names=[s.strip() for s in (self.root/f'{split}_list.txt').read_text().splitlines() if s.strip()]
        if len(set(x.casefold() for x in self.names))!=len(self.names): raise ValueError('duplicate split entries')
        for name in self.names:
            for d,key in self.paths(name):
                if key.casefold() not in self.files[d]: raise FileNotFoundError(f'missing {d}/{key}')
    def paths(self,name):
        stem=Path(name).stem
        return [('RGB',name),('Masks',stem+'.png')]+[('Multispectral',stem+'_'+c[3:].upper()+'.tif') for c in self.channels[3:]]
    def __len__(self): return len(self.names)
    def __getitem__(self,index):
        name=self.names[index]; files=[self.files[d][key.casefold()] for d,key in self.paths(name)]
        with Image.open(files[0]) as img:
            rgb=np.asarray(img.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32)/255
            original_size=img.size
        with Image.open(files[1]) as img:
            if img.size != original_size: raise ValueError(f'mask size mismatch: {files[1]}')
            if not np.isin(np.asarray(img),[0,255]).all(): raise ValueError(f'unexpected mask labels: {files[1]}')
            mask=np.asarray(img.resize((self.size,self.size),Image.Resampling.NEAREST))
        rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        channels=[rgb[:,:,i] for i in range(3)]
        for path in files[2:]:
            with Image.open(path) as img:
                if img.size != original_size: raise ValueError(f'band size mismatch: {path}')
                raw=np.asarray(img)
                if raw.ndim==3:
                    if raw.shape[2]!=3 or not (np.array_equal(raw[:,:,0],raw[:,:,1]) and np.array_equal(raw[:,:,0],raw[:,:,2])): raise ValueError(f'band channels differ: {path}')
                    raw=raw[:,:,0]
                if raw.ndim!=2: raise ValueError(f'unsupported band shape: {path}')
                scaled=Image.fromarray(raw).resize((self.size,self.size),Image.Resampling.BILINEAR)
                channels.append(np.asarray(scaled,dtype=np.float32)/float(self.config['ms_scale']))
        image=np.stack(channels)
        if self.split=='train' and self.config.get('augment_flip',False):
            if random.random()<.5: image=image[:,:,::-1]; mask=mask[:,::-1]
            if random.random()<.5: image=image[:,::-1,:]; mask=mask[::-1,:]
        return {'image':torch.from_numpy(image.copy()),'mask':torch.from_numpy((mask==255).astype(np.float32)[None].copy()),'sample_id':name,'dataset_id':'weedyrice','split':self.split,'input_spec':','.join(self.channels),'metadata':{'filename':name,'original_size':original_size}}
