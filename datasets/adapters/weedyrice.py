import csv
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets.registry import register
from tools.inspect_dataset import indexed_files

MODES = {'rgb': ('rgb_r','rgb_g','rgb_b'), 'rgbnir4': ('rgb_r','rgb_g','rgb_b','ms_nir'), 'rgbms7': ('rgb_r','rgb_g','rgb_b','ms_g','ms_r','ms_re','ms_nir')}

@register('weedyrice')
class WeedyRice(Dataset):
    def __init__(self, split, config):
        self.root=Path(config['root']); self.split=split; self.config=config
        self.mode=config.get('input_mode','rgb')
        if self.mode not in MODES: raise ValueError(f'unknown input mode {self.mode}')
        self.channels=tuple(config.get('channels',{}).get(self.mode, MODES[self.mode]))
        if len(self.channels)!=len(MODES[self.mode]) or set(self.channels)!=set(MODES[self.mode]):
            raise ValueError(f'channels for {self.mode} must contain exactly {MODES[self.mode]}')
        size=config['size']; size=(size,size) if isinstance(size,int) else tuple(size)
        if len(size)!=2 or any(not isinstance(x,int) or x<=0 for x in size): raise ValueError('size must be positive int or [height,width]')
        self.size=size
        self.mean=np.asarray(config.get('rgb_mean',[.485,.456,.406]),dtype=np.float32)
        self.std=np.asarray(config.get('rgb_std',[.229,.224,.225]),dtype=np.float32)
        if self.mean.shape!=(3,) or self.std.shape!=(3,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any(): raise ValueError('invalid RGB normalization')
        self.scale=float(config.get('ms_scale',float('nan')))
        if any(c.startswith('ms_') for c in self.channels) and (not math.isfinite(self.scale) or self.scale<=0): raise ValueError('ms_scale must be finite and positive')
        self.files={d:indexed_files(self.root/d) for d in ('RGB','Masks','Multispectral')}
        self.names=[s.strip() for s in (self.root/f'{split}_list.txt').read_text().splitlines() if s.strip()]
        if len(set(x.casefold() for x in self.names))!=len(self.names): raise ValueError('duplicate split entries')
        for name in self.names:
            for d,key in self.paths(name):
                if key.casefold() not in self.files[d]: raise FileNotFoundError(f'missing {d}/{key}')
        self.metadata={}
        mapping=self.root/'Metadata/filename_mapping.csv'; metadata=self.root/'Metadata/image_metadata.csv'
        if mapping.exists() and metadata.exists():
            with metadata.open(newline='') as f: records={r['original_filename']:r for r in csv.DictReader(f)}
            with mapping.open(newline='') as f:
                for r in csv.DictReader(f):
                    if r['original_filename'] in records: self.metadata[r['standardized_filename'].casefold()]=records[r['original_filename']]
    def paths(self,name):
        stem=Path(name).stem
        return [('RGB',name),('Masks',stem+'.png')]+[('Multispectral',stem+'_'+c[3:].upper()+'.tif') for c in self.channels if c.startswith('ms_')]
    def __len__(self): return len(self.names)
    def __getitem__(self,index):
        name=self.names[index]; files=[self.files[d][key.casefold()] for d,key in self.paths(name)]
        out_size=(self.size[1],self.size[0])
        with Image.open(files[0]) as img:
            rgb=np.asarray(img.convert('RGB').resize(out_size,Image.Resampling.BILINEAR),dtype=np.float32)/255
            original_size=img.size
        with Image.open(files[1]) as img:
            if img.size != original_size: raise ValueError(f'mask size mismatch: {files[1]}')
            if not np.isin(np.asarray(img),[0,255]).all(): raise ValueError(f'unexpected mask labels: {files[1]}')
            mask=np.asarray(img.resize(out_size,Image.Resampling.NEAREST))
        rgb=(rgb-self.mean)/self.std
        values={c:rgb[:,:,i] for i,c in enumerate(('rgb_r','rgb_g','rgb_b'))}
        ms=[c for c in self.channels if c.startswith('ms_')]
        for c,path in zip(ms,files[2:]):
            with Image.open(path) as img:
                if img.size != original_size: raise ValueError(f'band size mismatch: {path}')
                raw=np.asarray(img)
                if raw.ndim==3:
                    if raw.shape[2]!=3 or not (np.array_equal(raw[:,:,0],raw[:,:,1]) and np.array_equal(raw[:,:,0],raw[:,:,2])): raise ValueError(f'band channels differ: {path}')
                    raw=raw[:,:,0]
                if raw.ndim!=2: raise ValueError(f'unsupported band shape: {path}')
                values[c]=np.asarray(Image.fromarray(raw).resize(out_size,Image.Resampling.BILINEAR),dtype=np.float32)/self.scale
        image=np.stack([values[c] for c in self.channels])
        if self.split=='train' and self.config.get('augment_flip',False):
            if random.random()<.5: image=image[:,:,::-1]; mask=mask[:,::-1]
            if random.random()<.5: image=image[:,::-1,:]; mask=mask[::-1,:]
        metadata={'filename':name,'rgb_path':str(files[0]),'mask_path':str(files[1]),'foreground_values':[255],'original_size':original_size, **self.metadata.get(name.casefold(),{})}
        return {'image':torch.from_numpy(image.copy()),'mask':torch.from_numpy((mask==255).astype(np.float32)[None].copy()),'sample_id':name,'dataset_id':'weedyrice','split':self.split,'input_spec':','.join(self.channels),'metadata':metadata}
