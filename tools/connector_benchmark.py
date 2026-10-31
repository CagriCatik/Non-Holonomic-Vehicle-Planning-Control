"""Fixed-seed geometry sampling microbenchmark; no collision/renderer side effects."""
import json,sys,time
from pathlib import Path
import numpy as np
from non_holonomic_vehicle_planning import plan_dubins_segments,plan_reeds_shepp_segments
rng=np.random.default_rng(42)
poses=rng.uniform([-15,-15,-np.pi],[15,15,np.pi],size=(50,3))
rows=[];arrays={}
for name,connector in [('dubins',plan_dubins_segments),('reeds_shepp',plan_reeds_shepp_segments)]:
 start=time.perf_counter();points=0
 for i,(x,y,yaw) in enumerate(poses):
  paths=connector(0,0,0,x,y,yaw,[.1,.15,.2],step=.2)
  for j,path in enumerate(paths):
   arrays[f'{name}_{i}_{j}']=np.column_stack((path.x,path.y,path.yaw,path.direction))
   points+=len(path.x)
 rows.append(dict(connector=name,wall_time_s=time.perf_counter()-start,points=points,seed=42))
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
out.with_suffix('.json').write_text(json.dumps(rows,indent=2))
np.savez_compressed(out.with_suffix('.npz'),**arrays)
print(rows,flush=True)
