import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from datasets.registry import build_dataset
from engine.inference import load_model,load_pretrained,predict_arrays,error_overlay,preprocess_rgb,binary_mask
import yaml

def save(out,name,rgb,prob,mask,truth=None):
 folder=out/Path(name).stem;folder.mkdir(parents=True,exist_ok=True)
 Image.fromarray(rgb).save(folder/'rgb.png');Image.fromarray((mask*255).astype(np.uint8)).save(folder/'prediction.png');np.save(folder/'probability.npy',prob)
 if truth is not None:
  gt=(truth*255).astype(np.uint8);pr=(mask*255).astype(np.uint8);errors=error_overlay(rgb,truth,mask)
  Image.fromarray(gt).save(folder/'ground_truth.png');Image.fromarray(errors).save(folder/'errors.png')
  comparison=np.concatenate([rgb,np.repeat(gt[:,:,None],3,axis=2),np.repeat(pr[:,:,None],3,axis=2),errors],axis=1)
  Image.fromarray(comparison).save(folder/'comparison.png')

def main():
 p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True);source.add_argument('--checkpoint');source.add_argument('--pretrained-config');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--split',choices=['train','val','test']);p.add_argument('--input');p.add_argument('--limit',type=int);a=p.parse_args()
 if bool(a.split)==bool(a.input): p.error('choose exactly one of --split or --input')
 model,cfg=load_model(a.checkpoint,a.device) if a.checkpoint else load_pretrained(yaml.safe_load(Path(a.pretrained_config).read_text()),a.device);ds=cfg['dataset'];th=cfg['training'].get('threshold',.5);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 (out/'run.json').write_text(json.dumps({'checkpoint':a.checkpoint,'pretrained_config':a.pretrained_config,'split':a.split,'input':a.input,'config':cfg},indent=2)+'\n')
 if a.split:
  dataset=build_dataset(ds['name'],a.split,{**ds,'augment_flip':False})
  for i in range(min(len(dataset),a.limit or len(dataset))):
   sample=dataset[i];path=sample['metadata']['rgb_path'];maskpath=sample['metadata']['mask_path']
   with Image.open(path) as im: rgb=np.asarray(im.convert('RGB'))
   with Image.open(maskpath) as im: truth=binary_mask(np.asarray(im),sample['metadata']['foreground_values'])
   prob,pred=predict_arrays(model,sample['image'][None],a.device,rgb.shape[:2],th);save(out,sample['sample_id'],rgb,prob,pred,truth)
 else:
  if ds['input_mode']!='rgb': raise ValueError('multispectral checkpoints require dataset --split with real band files')
  source=Path(a.input); paths=[source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
  for path in paths[:a.limit]:
   with Image.open(path) as im: rgb=np.asarray(im.convert('RGB'));image=preprocess_rgb(im,ds)
   prob,pred=predict_arrays(model,image,a.device,rgb.shape[:2],th);save(out,path.name,rgb,prob,pred)
if __name__=='__main__':main()
