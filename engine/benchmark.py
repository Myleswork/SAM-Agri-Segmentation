import numpy as np

def latency_summary(milliseconds,batch_size):
    values=np.asarray(milliseconds,dtype=float)
    if values.size==0 or np.any(values<=0): raise ValueError('latencies must be positive')
    mean=float(values.mean())
    return {'batch_size':batch_size,'mean_ms':mean,'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),'images_per_second':batch_size*1000/mean}
