"""Explicit random-weight GPU integration smoke; no long training."""
import argparse, gc, json, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from datasets.registry import build_dataset
from engine.losses import structure_loss
from engine.checkpoint import save_checkpoint,load_checkpoint
from models.factory import build_model,SegmentationModel,parameter_summary

CHANNELS=['rgb_r','rgb_g','rgb_b','ms_g','ms_r','ms_re','ms_nir']

def smoke(name,size,root):
    cfg={'root':root,'input_mode':'rgbms7','size':size,'ms_scale':255.0}
    dataset=build_dataset('weedyrice','train',cfg); sample=dataset[0]
    x=sample['image'].unsqueeze(0).cuda(); target=sample['mask'].unsqueeze(0).cuda()
    model=build_model(name,CHANNELS,{'size':size,'device':'cuda:0'}); model.eval()
    shapes={}
    with torch.inference_mode():
        for n,ix in [('rgb',[0,1,2]),('rgbnir4',[0,1,2,6]),('rgbms7',list(range(7)))]:
            wrapper=SegmentationModel(model.network,[CHANNELS[i] for i in ix]).cuda().eval()
            out=wrapper(x[:,ix])['logits']; shapes[n]={'shape':list(out.shape),'finite':bool(torch.isfinite(out).all())}
            del wrapper,out
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001)
    trainable={id(p) for p in model.parameters() if p.requires_grad}; included={id(p) for g in opt.param_groups for p in g['params']}
    if trainable!=included: raise RuntimeError('optimizer parameter mismatch')
    model.train(); steps=[]
    for image in (torch.randn_like(x),x):
        opt.zero_grad(set_to_none=True); before=model.stem.projection.weight.detach().clone(); result=model(image)
        loss=sum(structure_loss(o,target) for o in [result['logits'],*result['aux']]); loss.backward()
        grad=model.stem.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all(): raise RuntimeError('invalid stem gradient')
        opt.step(); delta=(model.stem.projection.weight.detach()[:,3:]-before[:,3:]).abs().max().item()
        if delta==0: raise RuntimeError('no spectral weight update')
        steps.append({'loss':float(loss.detach()),'spectral_max_update':delta})
    model.eval()
    with torch.inference_mode(): expected=model(x)['logits'].cpu()
    with tempfile.TemporaryDirectory(dir='results') as d:
        path=Path(d)/'checkpoint.pth'; save_checkpoint(path,model,opt,None,{'dataset':cfg,'model':name,'channels':CHANNELS,'weight_status':'random initialization'},2,0)
        with torch.no_grad(): model.stem.projection.weight.zero_()
        load_checkpoint(path,model,opt)
        with torch.inference_mode(): actual=model(x)['logits'].cpu()
        error=(actual-expected).abs().max().item()
        if error>1e-5: raise RuntimeError(f'checkpoint inference mismatch {error}')
    return {'model':name,'weight_status':'random initialization','shapes':shapes,'training_steps':steps,'checkpoint_max_error':error,'parameters':parameter_summary(model)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sam2_unet','sam3_unet'],required=True);p.add_argument('--root',default='/root/train/dataset/weedy-rice/extracted/WeedyRice-RGBMS-DB'); p.add_argument('--output',required=True);args=p.parse_args()
    result=smoke(args.model,352 if args.model=='sam2_unet' else 336,args.root);Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__': main()
