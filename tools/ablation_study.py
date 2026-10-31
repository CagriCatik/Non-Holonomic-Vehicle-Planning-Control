"""Controlled one-factor closed-loop ablations; full results, no animation."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
from non_holonomic_vehicle_planning import Scenario, PlannedPath, VehicleParams, MPCConfig, RealTimePlannerConfig, DynamicObstacle
from non_holonomic_vehicle_planning.control.kinematic_model import KinematicState
import non_holonomic_vehicle_planning.simulation.runner as module

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
base_class=module.KinematicState
cases=[('D_full',{}),('A_ideal',{'plant':'ideal'}),('B_pt1',{'plant':'pt1'}),
       ('C_rate',{'plant':'rate'}),('F_replanning',{'replanning':True}),('F_long',{'replanning':True,'time':40.}),
       ('G_dt_01',{'dt':.1}),('G_dt_03',{'dt':.3}),('H_speed_15',{'speed':1.5,'time':30.}),
       ('H_speed_35',{'speed':3.5}),('I_Rd_low',{'rd':.15}),('I_Rd_high',{'rd':5.}),
       ('I_Q_low',{'q':.5}),('I_R_high',{'r':2.})]
rows=[]
for name,opts in cases:
    if len(sys.argv)>2 and name not in sys.argv[2:]: continue
    class Plant(base_class):
        def update(self,*args,**kwargs):
            mode=opts.get('plant','full')
            original=self.vehicle
            if mode!='full':
                self.vehicle=copy.copy(original)
                if mode=='ideal':
                    self.vehicle.max_steer_rate=1e6
                    kwargs['use_actuator_dynamics']=False
                else:
                    self.vehicle.max_steer_accel=1e6
                    if mode=='pt1': self.vehicle.max_steer_rate=1e6
            try: return super().update(*args,**kwargs)
            finally: self.vehicle=original
    module.KinematicState=Plant
    cfg=MPCConfig(horizon=14,dt=opts.get('dt',.2),target_speed=opts.get('speed',2.5),time_max=opts.get("time",18.),iter_max=3,
                  Rd=np.diag([.02,opts.get('rd',1.5)]))
    cfg.Q*=opts.get('q',1.);cfg.R*=opts.get('r',1.)
    sc=Scenario(name,env_width=60,env_height=40,boundary_collision=False)
    sc.vehicle=VehicleParams()
    xs=np.linspace(5,35,151);ys=15+1.2*(1-np.cos((xs-5)*np.pi/15))
    yaw=np.arctan(1.2*np.pi/15*np.sin((xs-5)*np.pi/15))
    sc.set_planned_path(PlannedPath(xs,ys,yaw,np.ones(len(xs),dtype=int)))
    if opts.get('replanning'):
        sc.add_dynamic_obstacle(DynamicObstacle(id='lead',kind='vehicle',start=(18.,15.),yaw_deg=0.,speed=.2,
                                                waypoints=[(18.,15.),(25.,15.)]))
    print('START',name,flush=True)
    try:
        result=module.SimulationRunner(cfg,RealTimePlannerConfig(enabled=opts.get('replanning',False))).run(sc)
        common=[]
        for row in range(1,len(result.t)):
            points=np.column_stack((result.ref_horizon_x[row],result.ref_horizon_y[row]))
            ab=np.diff(points,axis=0); lengths=np.sum(ab*ab,axis=1)
            fraction=np.clip(np.sum((np.array([result.x[row],result.y[row]])-points[:-1])*ab,axis=1)/np.maximum(lengths,1e-12),0,1)
            projection=points[:-1]+fraction[:,None]*ab
            dist=np.sum((projection-[result.x[row],result.y[row]])**2,axis=1)
            k=int(np.argmin(dist))
            angle=np.arctan2(ab[k,1],ab[k,0]) if lengths[k]>1e-10 else result.yaw[row]-result.heading_error[row]
            error=np.array([result.x[row],result.y[row]])-projection[k]
            common.append(-error[0]*np.sin(angle)+error[1]*np.cos(angle))
        delta=result.steer; rate=np.diff(delta)/cfg.dt; acc=np.diff(rate)/cfg.dt
        rms=lambda a:float(np.sqrt(np.mean(np.square(a)))) if len(a) else 0.
        signs=np.sign(rate[np.abs(rate)>np.deg2rad(1.)])
        row=dict(case=name,options=opts,reached_goal=result.reached_goal,cte_rms=rms(result.lateral_error),
                 common_cte_rms=rms(common),cte_max=float(np.max(abs(result.lateral_error))),heading_rms=rms(result.heading_error),
                 steering_rms=rms(delta),steering_ptp=float(np.ptp(delta)),steering_tv=float(np.sum(abs(np.diff(delta)))),
                 rate_rms=rms(rate),rate_max=float(np.max(abs(rate))),accel_rms=rms(acc),
                 accel_max=float(np.max(abs(acc))),reversals=int(np.sum(signs[1:]!=signs[:-1])),
                 replans=result.replan_count,simulation_time=float(result.t[-1]),wall_time=result.wall_time_s)
        np.savez_compressed(out/(name+'.npz'),t=result.t,steer=delta,cte=result.lateral_error,common_cte=common,x=result.x,y=result.y,yaw=result.yaw)
        if hasattr(result,'export'):result.export(out/name)
    except Exception as e:
        row=dict(case=name,error=repr(e))
    rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2))
    print('DONE',row,flush=True)
module.KinematicState=base_class
