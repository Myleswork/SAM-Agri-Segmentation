import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from engine.trainer import train

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--stop-after-epoch',type=int);a=p.parse_args()
 config=yaml.safe_load(Path(a.config).read_text());output=Path(a.output)
 if output.exists() and any(output.iterdir()) and not a.resume: raise FileExistsError('output directory is not empty; choose a new run directory')
 output.mkdir(parents=True,exist_ok=True)
 if not a.resume: (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 train(config,output,a.resume,a.stop_after_epoch)
if __name__=='__main__':main()
