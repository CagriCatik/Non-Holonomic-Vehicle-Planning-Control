"""Pure trajectory geometry, feasibility diagnostics and signed speed profiling."""
import numpy as np


def geometry(x,y,yaw,reversing):
    x,y,yaw = map(lambda a:np.asarray(a,dtype=float),(x,y,yaw))
    rev = np.asarray(reversing,dtype=bool)
    if x.ndim != 1 or len(x)<2 or any(len(a)!=len(x) for a in (y,yaw,rev)):
        raise ValueError("Trajectory arrays must have equal length >= 2")
    if not np.all(np.isfinite(np.r_[x,y,yaw])):
        raise ValueError("Trajectory values must be finite")
    ds = np.hypot(np.diff(x),np.diff(y))
    s = np.r_[0.,np.cumsum(ds)]
    kappa = np.zeros(len(x))
    cuts = np.r_[0,np.flatnonzero(np.diff(rev))+1,len(x)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if b-a<2:
            continue
        # Duplicate points cannot enter a derivative stencil.
        local_s,indices = np.unique(s[a:b],return_index=True)
        if len(local_s)<2:
            continue
        heading = np.unwrap(yaw[a:b])[indices]
        deriv = np.gradient(heading,local_s)
        kappa[a:b] = np.interp(s[a:b],local_s,deriv)*(-1 if rev[a] else 1)
    return s,kappa


def profile_speed(x,y,yaw,reversing,target_speed,max_accel=1.5,max_lateral_accel=None,
                  max_steer_rate=None,wheelbase=2.9):
    """Forward/backward v² envelope in each gear; stop at cusps and final point.

    max_lateral_accel is optional and must be explicitly supplied in m/s².
    Steering slew imposes a speed envelope without filtering the path geometry.
    """
    if max_accel<=0 or target_speed<=0:
        raise ValueError("Acceleration and target speed must be positive")
    s,kappa = geometry(x,y,yaw,reversing)
    rev = np.asarray(reversing,dtype=bool)
    speeds = np.full(len(s),abs(target_speed))
    if max_lateral_accel is not None:
        if max_lateral_accel<=0:
            raise ValueError("Lateral acceleration must be positive")
        speeds = np.minimum(speeds,np.sqrt(max_lateral_accel/np.maximum(abs(kappa),1e-9)))
    cuts = np.r_[0,np.flatnonzero(np.diff(rev))+1,len(s)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if max_steer_rate is not None and b-a>1:
            ss,ii = np.unique(s[a:b],return_index=True)
            if len(ss)>1:
                steering = np.arctan(wheelbase*kappa[a:b])[ii]
                gradient = np.interp(s[a:b],ss,abs(np.gradient(steering,ss)))
                speeds[a:b] = np.minimum(speeds[a:b],max_steer_rate/np.maximum(gradient,1e-9))
        speeds[b-1] = 0.
        for j in range(b-2,a-1,-1):
            speeds[j] = min(speeds[j],np.sqrt(speeds[j+1]**2+2*max_accel*(s[j+1]-s[j])))
        # A new gear starts from rest at the preceding cusp, including the gap.
        if a>0:
            speeds[a] = min(speeds[a],np.sqrt(2*max_accel*(s[a]-s[a-1])))
        for j in range(a+1,b):
            speeds[j] = min(speeds[j],np.sqrt(speeds[j-1]**2+2*max_accel*(s[j]-s[j-1])))
    return np.where(rev,-speeds,speeds)


def quintic_blend(u):
    """Unit offset with zero first and second derivative at both boundaries."""
    u = np.clip(u,0.,1.)
    return u**3*(10.+u*(-15.+6.*u))


def quintic_transition(s,length,d0,d1,d2,d_end):
    """d, d', d'' of the quintic with start (d0,d1,d2) and end (d_end,0,0) at s=length."""
    if length<=0:
        raise ValueError("Transition length must be positive")
    T = float(length)
    matrix = [[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]]
    rhs = [d_end-(d0+d1*T+.5*d2*T**2),-(d1+d2*T),-d2]
    poly = np.polynomial.Polynomial(np.r_[d0,d1,.5*d2,np.linalg.solve(matrix,rhs)])
    return poly(s),poly.deriv(1)(s),poly.deriv(2)(s)


def min_transition_length(offset,max_curvature):
    """Shortest rest-to-rest quintic whose peak |d''| = 10/sqrt(3)*|offset|/L² stays in limit."""
    return float(np.sqrt(10/np.sqrt(3)*abs(offset)/max_curvature))


def lateral_bypass_profile(s,s_pull,s_pass,s_end,offset,d0=0.,d1=0.,d2=0.):
    """Frenet offset d(s), d'(s): pull-out from the measured state, hold, merge to d=0.

    The profile is analytic, so measured start errors enter as boundary values
    instead of being amplified by interpolating densely spaced path samples.
    C² holds at both phase junctions; samples at s<0 hold the start value.
    """
    s = np.asarray(s,dtype=float)
    d = np.full(s.shape,float(offset))
    slope = np.zeros(s.shape)
    pull = s<=s_pull
    d[pull],slope[pull],_ = quintic_transition(np.maximum(s[pull],0.),s_pull,d0,d1,d2,offset)
    merge = s>s_pass
    d[merge],slope[merge],_ = quintic_transition(s[merge]-s_pass,s_end-s_pass,offset,0.,0.,0.)
    return d,slope


def inspect_path(path,vehicle):
    """Sampled feasibility diagnostics; this is not a swept-volume certificate."""
    s,kappa=geometry(path.x,path.y,path.yaw,path.reversing)
    ds=np.diff(s)
    tangent=np.arctan2(np.diff(path.y),np.diff(path.x))
    vehicle_tangent=path.yaw[:-1]+np.where(path.reversing[:-1],np.pi,0.)
    errors=(tangent-vehicle_tangent+np.pi)%(2*np.pi)-np.pi
    valid=(ds>1e-8)&(path.direction[:-1]==path.direction[1:])
    physical=np.tan(vehicle.max_steer)/vehicle.wheelbase
    return dict(length_m=float(s[-1]),max_curvature_m_inv=float(np.max(abs(kappa))),
                physical_curvature_limit_m_inv=float(physical),
                curvature_feasible=bool(np.max(abs(kappa))<=physical+1e-3),
                tangent_error_max_rad=float(np.max(abs(errors[valid]))) if np.any(valid) else 0.,
                duplicate_edges=int(np.sum(ds<=1e-8)),gear_changes=int(np.sum(np.diff(path.direction)!=0)))


def sample_primitive(distances,mode,curvature,origin):
    """Vectorized analytic Dubins/RS primitive, preserving signed arc length."""
    distances=np.asarray(distances)
    ox,oy,yaw=origin
    if mode=='S':
        return (ox+distances/curvature*np.cos(yaw),oy+distances/curvature*np.sin(yaw),
                np.full_like(distances,yaw))
    sign=1. if mode=='L' else -1.
    dx=np.sin(distances)/curvature
    dy=sign*(1-np.cos(distances))/curvature
    return ox+np.cos(yaw)*dx-np.sin(yaw)*dy,oy+np.sin(yaw)*dx+np.cos(yaw)*dy,yaw+sign*distances
