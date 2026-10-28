import numpy as np
import torch
import torch.nn.functional as F
from models.factory import build_model

def predict_arrays(model,image,device,output_size,threshold=.5):
    model.eval()
    with torch.inference_mode():
        logits=model(image.to(device))['logits'];logits=F.interpolate(logits,size=output_size,mode='bilinear',align_corners=False)
        prob=logits.sigmoid()[0,0].cpu().numpy()
    return prob,prob>=threshold

def error_overlay(rgb,truth,pred):
    out=np.asarray(rgb,dtype=np.float32)*.5
    truth=np.asarray(truth,dtype=bool);pred=np.asarray(pred,dtype=bool)
    for mask,color in [(truth&pred,(0,255,0)),(~truth&pred,(255,0,0)),(truth&~pred,(0,0,255))]: out[mask]+=np.array(color,dtype=np.float32)*.5
    return np.clip(out,0,255).astype(np.uint8)

def load_model(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);config=state['config'];mc=dict(config['model']);mc['device']=device;mc['pretrained']=None
    ds=config['dataset'];channels=ds['channels'][ds['input_mode']];model=build_model(mc['name'],channels,mc);model.load_state_dict(state['model'],strict=True);model.eval()
    torch.backends.cuda.matmul.allow_tf32=bool(config['training'].get('allow_tf32',False));torch.backends.cudnn.allow_tf32=bool(config['training'].get('allow_tf32',False))
    return model,config

def preprocess_rgb(image,config):
    from PIL import Image
    size=config['size']; size=(size,size) if isinstance(size,int) else size
    arr=np.asarray(image.convert('RGB').resize((size[1],size[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
    arr=(arr-np.asarray(config.get('rgb_mean',[.485,.456,.406]),np.float32))/np.asarray(config.get('rgb_std',[.229,.224,.225]),np.float32)
    channels=config['channels'][config['input_mode']]; indices=[('rgb_r','rgb_g','rgb_b').index(c) for c in channels]
    return torch.from_numpy(arr[:,:,indices].transpose(2,0,1).copy())[None]

def load_pretrained(config,device):
    mc=dict(config['model'])
    if not mc.get('pretrained'): raise ValueError('pretrained checkpoint path required')
    mc['device']=device;ds=config['dataset'];model=build_model(mc['name'],ds['channels'][ds['input_mode']],mc)
    model.eval();torch.backends.cuda.matmul.allow_tf32=bool(config.get('training',{}).get('allow_tf32',False));torch.backends.cudnn.allow_tf32=bool(config.get('training',{}).get('allow_tf32',False))
    config=dict(config);config['provenance']={'weight_status':'pretrained encoder; untrained segmentation head'}
    return model,config

def binary_mask(mask,foreground_values):
    if not foreground_values: raise ValueError('adapter must supply foreground_values')
    if np.asarray(mask).ndim!=2: raise ValueError('visualization requires a two-dimensional label mask')
    return np.isin(np.asarray(mask),foreground_values)
