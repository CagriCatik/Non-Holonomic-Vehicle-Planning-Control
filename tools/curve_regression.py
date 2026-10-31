"""Deterministic high-curvature S bend, without RRT/planner timing confounds."""
import sys
from pathlib import Path
import numpy as np
from non_holonomic_vehicle_planning import Scenario,PlannedPath,SimulationRunner,MPCConfig,RealTimePlannerConfig

def make_curve():
    n=81; angle=np.linspace(0,np.pi/2,n)
    x=np.r_[np.linspace(5,10,51)[:-1],10+5*np.sin(angle),15+5*(1-np.cos(angle[1:])),np.linspace(20,25,51)[1:]]
    y=np.r_[np.full(50,15),15+5*(1-np.cos(angle)),20+5*np.sin(angle[1:]),np.full(50,25)]
    yaw=np.r_[np.zeros(50),angle,np.pi/2-angle[1:],np.zeros(50)]
    return PlannedPath(x,y,yaw,np.ones(len(x),dtype=int))

if __name__=='__main__':
    sc=Scenario('S_bend',env_width=40,env_height=40)
    sc.set_planned_path(make_curve())
    cfg=MPCConfig(horizon=14,time_max=30,target_speed=2.5,iter_max=4,Rd=np.diag([.02,1.5]))
    r=SimulationRunner(cfg,RealTimePlannerConfig(enabled=False)).run(sc)
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(out)+'.npz',t=r.t,x=r.x,y=r.y,steer=r.steer,cte=r.lateral_error)
    if hasattr(r,'export'):r.export(out)
    print(r.reached_goal,float(np.max(abs(r.lateral_error))),float(np.sqrt(np.mean(r.lateral_error**2))),flush=True)
