"""Explicit real-data one-step GPU smoke for RGB and RGB+NIR."""
import gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from datasets.registry import build_dataset
from engine.losses import structure_loss
from models.factory import build_model

def main():
  root='/root/train/dataset/weedy-rice/extracted/WeedyRice-RGBMS-DB'; results=[]
  for name,size in [('sam2_unet',352),('sam3_unet',336)]:
    for mode,channels in [('rgb',['rgb_r','rgb_g','rgb_b']),('rgbnir4',['rgb_r','rgb_g','rgb_b','ms_nir'])]:
      ds=build_dataset('weedyrice','train',{'root':root,'size':size,'input_mode':mode,'ms_scale':255}); sample=ds[0]
      m=build_model(name,channels,{'size':size,'device':'cuda:0'});m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
      before=m.stem.projection.weight.detach().clone() if mode!='rgb' else None
      out=m(sample['image'][None].cuda());loss=sum(structure_loss(t,sample['mask'][None].cuda()) for t in [out['logits'],*out['aux']]);loss.backward()
      gradients=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
      if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise RuntimeError('invalid gradients')
      opt.step();delta=(m.stem.projection.weight.detach()[:,3:]-before[:,3:]).abs().max().item() if before is not None else None
      if delta is not None and not delta>0:raise RuntimeError('spectral parameter did not change')
      results.append({'model':name,'mode':mode,'shape':list(out['logits'].shape),'loss':float(loss.detach()),'spectral_update':delta,'weight_status':'random initialization'});print(results[-1],flush=True)
      del m,opt,out,loss,gradients;gc.collect();torch.cuda.empty_cache()
  Path('results/smoke_modalities.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
