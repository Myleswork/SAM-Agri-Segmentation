"""Binary metrics. Undefined ratios are NaN and excluded from macro means."""
import math
import torch

def confusion_metrics(tp,fp,fn,tn):
    def ratio(a,b): return a/b if b else float('nan')
    fg=ratio(tp,tp+fp+fn); bg=ratio(tn,tn+fp+fn)
    valid=[v for v in (fg,bg) if not math.isnan(v)]
    return {'foreground_iou':fg,'background_iou':bg,'miou':sum(valid)/len(valid) if valid else float('nan'),'dice':ratio(2*tp,2*tp+fp+fn),'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn)}

def confusion_from_logits(logits,target,threshold=0.5):
    if logits.shape!=target.shape: raise ValueError('prediction and target shape mismatch')
    if not 0<=threshold<=1: raise ValueError('threshold must be in [0,1]')
    if not torch.all((target==0)|(target==1)): raise ValueError('binary target required')
    pred=torch.sigmoid(logits)>=threshold; truth=target.bool()
    return {'tp':int((pred & truth).sum()),'fp':int((pred & ~truth).sum()),'fn':int((~pred & truth).sum()),'tn':int((~pred & ~truth).sum())}

def aggregate_metrics(confusions):
    if not confusions: raise ValueError('no images to evaluate')
    totals={k:sum(x[k] for x in confusions) for k in ('tp','fp','fn','tn')}
    rows=[confusion_metrics(**x) for x in confusions]
    macro={}
    for key in rows[0]:
        valid=[r[key] for r in rows if not math.isnan(r[key])]
        macro[key]=sum(valid)/len(valid) if valid else float('nan')
    return {'counts':totals,'micro':confusion_metrics(**totals),'macro':macro,'macro_valid_counts':{k:sum(not math.isnan(r[k]) for r in rows) for k in rows[0]}}
