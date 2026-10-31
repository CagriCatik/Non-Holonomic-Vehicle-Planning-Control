"""Reproducible telemetry export and physical, unfiltered tracking metrics."""
import csv
import json
from pathlib import Path
import numpy as np


def rms(values):
    values=np.asarray(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.


def summarize(result):
    dt=np.diff(result.t)
    rate=np.diff(result.steer)/dt
    angular_acceleration=np.diff(rate)/dt[1:]
    significant=np.sign(rate[abs(rate)>np.deg2rad(1.)])
    settling=None
    # Explicit definition: after this instant |CTE| <= 0.2 m and |heading| <= 5 deg
    # for the rest of the recorded run, provided the final 2 s satisfy both.
    good=(abs(result.lateral_error)<=.2)&(abs(result.heading_error)<=np.deg2rad(5.))
    bad=np.flatnonzero(~good)
    candidate=int(bad[-1]+1) if len(bad) else 0
    if candidate<len(good) and result.t[-1]-result.t[candidate]>=2.:
        settling=float(result.t[candidate])
    return dict(scenario=result.scenario_name,reached_goal=bool(result.reached_goal),termination_reason=result.termination_reason,
        simulation_time_s=float(result.t[-1]),wall_time_s=result.wall_time_s,
        cte_rms_m=rms(result.lateral_error),cte_max_m=float(np.max(abs(result.lateral_error))),
        heading_rms_rad=rms(result.heading_error),heading_max_rad=float(np.max(abs(result.heading_error))),
        steering_rms_rad=rms(result.steer),steering_ptp_rad=float(np.ptp(result.steer)),
        steering_tv_rad=float(np.sum(abs(np.diff(result.steer)))),
        steering_reversals=int(np.sum(significant[1:]!=significant[:-1])),
        steering_rate_rms_rad_s=rms(rate),steering_rate_max_rad_s=float(np.max(abs(rate))) if len(rate) else 0.,
        steering_accel_rms_rad_s2=rms(angular_acceleration),
        steering_accel_max_rad_s2=float(np.max(abs(angular_acceleration))) if len(angular_acceleration) else 0.,
        tracking_settling_time_s=settling,replan_count=result.replan_count,
        solver_fallback_count=int(np.sum(result.telemetry.get('solver_fallback',[]))),
        min_dynamic_clearance_m=float(np.min(result.min_dynamic_clearances_m)),
        min_static_clearance_m=float(np.min(result.telemetry.get('static_clearance_m',[float('inf')]))))


def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.generic): return value.item()
    return value


def export_result(result,destination):
    stem=Path(destination);stem.parent.mkdir(parents=True,exist_ok=True)
    fields=dict(time_s=result.t,x_m=result.x,y_m=result.y,yaw_rad=result.yaw,speed_m_s=result.v,
                steering_actual_rad=result.steer,cte_m=result.lateral_error,heading_error_rad=result.heading_error,
                planner_mode=result.planner_modes,dynamic_clearance_m=result.min_dynamic_clearances_m,
                replan_latency_ms=result.replan_latencies_ms,**result.telemetry)
    if any(len(v)!=len(result.t) for v in fields.values()):
        raise ValueError('Telemetry arrays are not time-aligned')
    csv_path=Path(str(stem)+'.csv');json_path=Path(str(stem)+'.json')
    with csv_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(fields)
        writer.writerows([[clean(v) for v in row] for row in zip(*fields.values())])
    payload=dict(schema_version=1,units='SI',sample_convention='state at time_s; commands applied over preceding interval',
                 metrics=clean(summarize(result)))
    json_path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return csv_path,json_path
