def confusion_metrics(tp,fp,fn,tn):
    def ratio(a,b): return a/b if b else 1.0
    fg=ratio(tp,tp+fp+fn); bg=ratio(tn,tn+fp+fn)
    return {'foreground_iou':fg,'background_iou':bg,'miou':(fg+bg)/2,'dice':ratio(2*tp,2*tp+fp+fn),'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn)}
