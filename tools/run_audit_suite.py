"""Run isolated headless scenario processes with explicit budgets and concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys
from pathlib import Path
from non_holonomic_vehicle_planning.scenarios.definitions import SCENARIO_SPECS

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output')
parser.add_argument('--path-cache',type=Path)
parser.add_argument('--workers',type=int,default=4)
parser.add_argument('--timeout-seconds',type=float,default=600.)
args=parser.parse_args()
if args.workers<1:parser.error('workers must be positive')
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
(out/'suite.json').write_text(json.dumps(dict(workers=args.workers,wall_budget_s=args.timeout_seconds,seed=42),indent=2))
script=Path(__file__).with_name('audit_benchmark.py').resolve()

def run(key):
    destination=out/key
    with (out/(key+'.log')).open('w') as log:
        command=[sys.executable,str(script),str(destination),key,'--timeout-seconds',str(args.timeout_seconds)]
        if args.path_cache is not None:command += ['--path-cache',str(args.path_cache)]
        process=subprocess.run(command,
                               stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    summary=destination/'summary.json'
    if summary.exists():return json.loads(summary.read_text())[0]
    return dict(scenario=key,error=f'Worker failed with return code {process.returncode}')

rows=[]
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    keys=[key for key in SCENARIO_SPECS if args.path_cache is None or (args.path_cache/key/f'{key}_initial_path.npz').exists()]
    futures=[executor.submit(run,key) for key in keys]
    for future in as_completed(futures):
        row=future.result();rows.append(row)
        (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        print(row['scenario'],row.get('reached_goal',row.get('error')),flush=True)
