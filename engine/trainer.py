import csv,json,logging,math,random,platform,subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.registry import build_dataset
from engine.checkpoint import load_checkpoint,save_checkpoint
from engine.evaluator import evaluate
from engine.losses import structure_loss
from engine.serialization import json_safe
from models.factory import build_model,parameter_summary,COMMITS

def train(config,output,resume=None,stop_after_epoch=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('sam_agri_training'); logger.setLevel(logging.INFO); logger.propagate=False
    for old_handler in logger.handlers: old_handler.close()
    logger.handlers.clear()
    formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(),logging.FileHandler(output/'train.log')):
        handler.setFormatter(formatter); logger.addHandler(handler)
    settings=config['training']; ds=config['dataset']; mc=config['model']; seed=int(settings.get('seed',42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device=mc.get('device','cpu'); mode=ds.get('input_mode','rgb'); channels=ds['channels'][mode]
    model=build_model(mc['name'],channels,mc)
    torch.backends.cuda.matmul.allow_tf32=bool(settings.get('allow_tf32',False)); torch.backends.cudnn.allow_tf32=bool(settings.get('allow_tf32',False))
    logger.info('parameters: %s',json.dumps(parameter_summary(model)))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(settings['lr']),weight_decay=float(settings.get('weight_decay',0)))
    expected={id(p) for p in model.parameters() if p.requires_grad}; actual={id(p) for g in optimizer.param_groups for p in g['params']}
    if expected!=actual: raise RuntimeError('optimizer parameter mismatch')
    epochs=int(settings['epochs'])
    if epochs<1: raise ValueError('epochs must be positive')
    for k in ('max_train_batches','max_eval_batches'):
        if settings.get(k) is not None and int(settings[k])<1: raise ValueError(f'{k} must be positive or null')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    config['provenance']={'upstream_commit':COMMITS[mc['name']],'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,'git_head':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'working_tree_dirty':bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.strip()),'started_utc':datetime.now(timezone.utc).isoformat(),'weight_status':'pretrained' if mc.get('pretrained') else 'random initialization','channels':channels}
    start=0;best=-1
    if resume:
        state=load_checkpoint(resume,model,optimizer,scheduler,True)
        if state['config']['dataset'] != ds or state['config']['model'] != mc: raise ValueError('resume config does not match checkpoint')
        start=state['epoch'];best=state['best']
    loaders={split:DataLoader(build_dataset(ds['name'],split,ds),batch_size=int(settings['batch_size']),shuffle=split=='train',num_workers=int(settings.get('num_workers',0))) for split in ('train','val','test')}
    precision=settings.get('precision','fp32');
    if precision not in ('fp32','bf16'): raise ValueError('precision must be fp32 or bf16')
    if precision=='bf16' and not str(device).startswith('cuda'): raise ValueError('bf16 training requires cuda')
    accum=int(settings.get('grad_accumulation',1));
    if accum<1: raise ValueError('grad_accumulation must be >= 1')
    logpath=output/'epochs.csv'; new=not logpath.exists()
    with logpath.open('a',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['epoch','loss','val_fg_iou','val_dice','lr']);
        if new: writer.writeheader()
        end_epoch=min(epochs,int(stop_after_epoch)) if stop_after_epoch is not None else epochs
        if end_epoch<start: raise ValueError('stop_after_epoch precedes checkpoint epoch')
        for epoch in range(start,end_epoch):
            model.train(); optimizer.zero_grad(set_to_none=True); n=0; loss_sum=0.; cap=min(len(loaders['train']),settings.get('max_train_batches') or len(loaders['train']))
            for i,batch in enumerate(loaders['train']):
                if i>=cap: break
                x=batch['image'].to(device); target=batch['mask'].to(device)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                    out=model(x); loss=sum(structure_loss(o,target) for o in [out['logits'],*out['aux']])
                group=min(accum,cap-(i//accum)*accum); (loss/group).backward()
                if (i+1)%accum==0 or i+1==cap: optimizer.step();optimizer.zero_grad(set_to_none=True)
                n+=1;loss_sum+=loss.item();logger.info('epoch=%d/%d batch=%d/%d loss=%.6f',epoch+1,epochs,i+1,cap,loss.item())
            metrics,_=evaluate(model,loaders['val'],device,settings.get('threshold',.5),settings.get('max_eval_batches'))
            score=metrics['micro']['foreground_iou']; logger.info('validation %s',json.dumps(metrics['micro']))
            scheduler.step()
            if math.isfinite(score) and score>best:
                best=score;save_checkpoint(output/'best.pth',model,optimizer,scheduler,config,epoch+1,best)
            save_checkpoint(output/'last.pth',model,optimizer,scheduler,config,epoch+1,best)
            writer.writerow({'epoch':epoch+1,'loss':loss_sum/n,'val_fg_iou':score,'val_dice':metrics['micro']['dice'],'lr':optimizer.param_groups[0]['lr']});file.flush()
    if not (output/'best.pth').exists(): raise RuntimeError('no valid best checkpoint')
    load_checkpoint(output/'best.pth',model); test,rows=evaluate(model,loaders['test'],device,settings.get('threshold',.5),settings.get('max_eval_batches'))
    (output/'metrics.json').write_text(json.dumps(json_safe({'weight_status':config['provenance']['weight_status'],'test':test}),indent=2,allow_nan=False)+'\n')
    with (output/'per_image_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    return test
