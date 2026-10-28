import math
import torch
import torch.nn.functional as F
from metrics.segmentation import confusion_from_logits,confusion_metrics,aggregate_metrics

def evaluate(model,loader,device,threshold=.5,max_batches=None,coverage_edges=(0,.05,.1,.2,.4,.6,1.0)):
    if len(coverage_edges)<2 or coverage_edges[0]!=0 or coverage_edges[-1]!=1 or any(a>=b for a,b in zip(coverage_edges,coverage_edges[1:])): raise ValueError('coverage edges must increase from 0 to 1')
    model.eval(); rows=[]; confusions=[]; groups={f'{a:g}-{b:g}':[] for a,b in zip(coverage_edges,coverage_edges[1:])}
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            if max_batches is not None and i>=max_batches: break
            x=batch['image'].to(device); y=batch['mask'].to(device)
            logits=model(x)['logits']
            if logits.shape[-2:]!=y.shape[-2:]: logits=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
            for j in range(len(x)):
                c=confusion_from_logits(logits[j:j+1],y[j:j+1],threshold); coverage=float(y[j].float().mean()); confusions.append(c)
                row={'sample_id':batch['sample_id'][j],'coverage':coverage,**c,**confusion_metrics(**c)}; rows.append(row)
                for index,(a,b) in enumerate(zip(coverage_edges,coverage_edges[1:])):
                    if a<=coverage<b or (index==len(coverage_edges)-2 and coverage==b): groups[f'{a:g}-{b:g}'].append(c); break
    result=aggregate_metrics(confusions); result['coverage_groups']={k:{'count':len(v),'metrics':aggregate_metrics(v) if v else None} for k,v in groups.items()}
    return result,rows
