"""Headless fixed-seed catalog benchmark. Run with PYTHONPATH=src python tools/audit_benchmark.py OUT."""
import json
import hashlib
from collections import Counter
import subprocess
import sys
import argparse
import signal
import time
import traceback
import warnings
from pathlib import Path
import numpy as np
from non_holonomic_vehicle_planning.config import load_config, get_configs_dir
from non_holonomic_vehicle_planning.mpc import parse_mpc_config
from non_holonomic_vehicle_planning.scenarios.definitions import SCENARIO_SPECS, load_or_plan_scenario
from non_holonomic_vehicle_planning.simulation.runner import SimulationRunner

def strict(value):
    if isinstance(value,dict): return {k:strict(v) for k,v in value.items()}
    if isinstance(value,list): return [strict(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output')
parser.add_argument('scenarios',nargs='*')
parser.add_argument('--path-cache',type=Path,help='Replay initial paths from an earlier suite; skip RRT')
parser.add_argument('--timeout-seconds',type=float,default=600.,help='Explicit wall budget per scenario (planning + simulation)')
args=parser.parse_args()
if args.timeout_seconds<=0: parser.error('timeout must be positive')
def timeout_handler(*_):
    raise TimeoutError(f'Scenario wall budget {args.timeout_seconds:g} s exceeded')
signal.signal(signal.SIGALRM,timeout_handler)
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
metadata = {'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), 'seed':42,
            'python':sys.version, 'dependencies':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
root = Path(__import__('non_holonomic_vehicle_planning').__file__).parent
metadata['source_sha256'] = hashlib.sha256(b''.join(p.relative_to(root).as_posix().encode()+p.read_bytes() for p in sorted(root.rglob('*.py')))).hexdigest()
metadata['command'] = sys.argv
metadata['source_root'] = str(root)
(out/'environment.json').write_text(json.dumps(metadata, indent=2))
rows=[]
for key,spec in SCENARIO_SPECS.items():
    if args.scenarios and key not in args.scenarios: continue
    start=time.perf_counter(); row={'scenario':key,'seed':42,'wall_budget_s':args.timeout_seconds}
    signal.setitimer(signal.ITIMER_REAL,args.timeout_seconds)
    print('START',key,flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            app=load_config(get_configs_dir()/'scenarios'/spec.yaml_file if spec.yaml_file else None)
            cfg=parse_mpc_config(app.raw)
            row['phase']='planning'
            (out/'progress.json').write_text(json.dumps(row,indent=2))
            if args.path_cache is None:
                sc,_=load_or_plan_scenario(key,replan=True,seed=42)
            else:
                from non_holonomic_vehicle_planning.scenarios.definitions import _apply_configuration
                from non_holonomic_vehicle_planning.planning.base import PlannedPath
                sc=spec.factory()
                _apply_configuration(sc,app)
                path_file=args.path_cache/key/f'{key}_initial_path.npz'
                if not path_file.exists(): path_file=args.path_cache/f'{key}_initial_path.npz'
                with np.load(path_file) as archive:
                    sc.set_planned_path(PlannedPath(archive['x'],archive['y'],archive['yaw'],archive['direction']))
                row['cached_path_sha256']=hashlib.sha256(path_file.read_bytes()).hexdigest()
            planned=sc.planned_path
            np.savez_compressed(out/f'{key}_initial_path.npz',x=planned.x,y=planned.y,yaw=planned.yaw,direction=planned.direction)
            planning_wall=time.perf_counter()-start
            row['phase']='simulation'
            (out/'progress.json').write_text(json.dumps(row,indent=2))
            r=SimulationRunner(cfg,app.realtime).run(sc)
            row['phase']='complete' 
            dt=cfg.dt; rate=np.diff(r.steer)/dt; acc=np.diff(rate)/dt
            rms=lambda a:float(np.sqrt(np.mean(np.square(a)))) if len(a) else 0.
            row.update(planning_wall_s=planning_wall,simulation_wall_s=r.wall_time_s,termination_reason=getattr(r,"termination_reason","legacy_unspecified"),reached_goal=bool(r.reached_goal),simulation_time=float(r.t[-1]),
                cte_rms=rms(r.lateral_error),cte_max=float(np.max(np.abs(r.lateral_error))),
                heading_rms=rms(r.heading_error),steering_rms=rms(r.steer),steering_ptp=float(np.ptp(r.steer)),
                steering_tv=float(np.sum(np.abs(np.diff(r.steer)))),rate_rms=rms(rate),accel_rms=rms(acc),
                replans=r.replan_count,min_dynamic_clearance=float(np.min(r.min_dynamic_clearances_m)))
            np.savez_compressed(out/f'{key}.npz',t=r.t,x=r.x,y=r.y,yaw=r.yaw,v=r.v,steer=r.steer,
                accel=r.accel,cte=r.lateral_error,heading=r.heading_error,ref_x=r.ref_x,ref_y=r.ref_y,ref_yaw=r.ref_yaw,ref_reversing=r.ref_reversing)
            if hasattr(r,'export'):
                from non_holonomic_vehicle_planning.trajectory import inspect_path
                row['path_feasibility']=inspect_path(sc.planned_path,sc.vehicle)
                row['solver_fallbacks']=int(np.sum(r.telemetry['solver_fallback']))
                r.export(out/key)
        except Exception:
            row['error']=traceback.format_exc()
        row['warnings']=sorted(set(str(w.message) for w in caught))
        row['warning_counts']=dict(Counter(str(w.message) for w in caught))
    signal.setitimer(signal.ITIMER_REAL,0)
    row['wall_time']=time.perf_counter()-start
    rows.append(row)
    (out/'summary.json').write_text(json.dumps(strict(rows),indent=2,allow_nan=False))
    print('DONE',key,json.dumps({k:v for k,v in row.items() if k not in ('warnings','error')}),flush=True)
