import subprocess, sys, os
ROOT=os.path.dirname(__file__)+"/.."
def run(cmd): return subprocess.run(cmd, cwd=ROOT).returncode
def test_regression(): assert run([sys.executable,"-m","src.train","--task","regression","--train_csv","dummy_data/metadata.csv","--val_csv","dummy_data/metadata.csv","--target","egfr","--epochs","1","--batch_size","2"])==0
def test_classification(): assert run([sys.executable,"-m","src.train","--task","classification","--train_csv","dummy_data/metadata.csv","--val_csv","dummy_data/metadata.csv","--target","ckd_label","--epochs","1","--batch_size","2"])==0
def test_survival(): assert run([sys.executable,"-m","src.train","--task","survival","--train_csv","dummy_data/metadata_surv.csv","--val_csv","dummy_data/metadata_surv.csv","--target","time","--event_col","event","--epochs","1","--batch_size","2"])==0
