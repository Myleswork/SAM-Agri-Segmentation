"""One real-data BF16 training step per upstream model."""
import gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from datasets.registry import build_dataset
from models.factory import build_model
from engine.losses import structure_loss

def main():
 root='/root/train/dataset/weedy-rice/extracted/WeedyRice-RGBMS-DB';channels=['rgb_r','rgb_g','rgb_b','ms_g','ms_r','ms_re','ms_nir'];rows=[]
 for name,size in [('sam2_unet',352),('sam3_unet',336)]:
  sample=build_dataset('weedyrice','train',{'root':root,'input_mode':'rgbms7','size':size,'ms_scale':255})[0]
  model=build_model(name,channels,{'size':size,'device':'cuda:0'}).train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001)
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   out=model(sample['image'][None].cuda());loss=sum(structure_loss(t,sample['mask'][None].cuda()) for t in [out['logits'],*out['aux']])
  loss.backward();grad=model.stem.projection.weight.grad
  if not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all(): raise RuntimeError('non-finite BF16 loss or stem gradient')
  before=model.stem.projection.weight.detach().clone();opt.step();delta=(model.stem.projection.weight.detach()[:,3:]-before[:,3:]).abs().max().item()
  if delta<=0: raise RuntimeError('spectral weights did not update')
  rows.append({'model':name,'precision':'bf16','loss':loss.item(),'spectral_update':delta,'weights':'random initialization'});print(rows[-1],flush=True)
  del model,opt,out,loss,grad;gc.collect();torch.cuda.empty_cache()
 Path('results/smoke_bf16.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
