import os
import random
from pathlib import Path
import numpy as np
import torch

def save_checkpoint(path,model,optimizer,scheduler,config,epoch,best):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,'config':config,'epoch':epoch,'best':best,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
    temporary=path.with_suffix(path.suffix+'.tmp')
    try: torch.save(state,temporary); os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

def load_checkpoint(path,model,optimizer=None,scheduler=None,restore_rng=False):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    if optimizer is not None and state['optimizer'] is not None: optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and state['scheduler'] is not None: scheduler.load_state_dict(state['scheduler'])
    if restore_rng:
        torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
