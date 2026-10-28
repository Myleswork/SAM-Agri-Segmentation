import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from datasets.registry import build_dataset
from engine.evaluator import evaluate
from engine.inference import load_model
from engine.serialization import json_safe

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);p.add_argument('--max-batches',type=int);a=p.parse_args()
 model,config=load_model(a.checkpoint,a.device); ds=config['dataset'];loader=DataLoader(build_dataset(ds['name'],a.split,ds),batch_size=1,num_workers=0)
 metrics,rows=evaluate(model,loader,a.device,config['training'].get('threshold',.5),a.max_batches);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 (out/'metrics.json').write_text(json.dumps(json_safe({'split':a.split,'checkpoint':a.checkpoint,'weight_status':config.get('provenance',{}).get('weight_status','unknown'),'metrics':metrics}),indent=2,allow_nan=False)+'\n')
 with (out/'per_image_metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
 print(json.dumps(metrics['micro'],indent=2))
if __name__=='__main__':main()
