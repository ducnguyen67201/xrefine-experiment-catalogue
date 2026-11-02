import importlib.util
from pathlib import Path
import math
import pytest as pt
@pt.fixture(scope="module")
def a():
 import numpy as np
 p=Path(__file__).with_name("candidate.py")
 s=importlib.util.spec_from_file_location("px_fixture",p)
 c=importlib.util.module_from_spec(s)
 s.loader.exec_module(c)
 return np,c
def rot(np,j,a):
 c,s=math.cos(a),math.sin(a)
 if j==0:
  return np.array([[1.,0,0],[0,c,-s],[0,s,c]])
 if j==1:
  return np.array([[c,0,s],[0,1.,0],[-s,0,c]])
 return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def data(np,eq=False):
 X=np.array([[(-1)**i*(.2+i*.07),(i%4-.8)*.23,3.5+(i%5)*.31]for i in range(12)])
 Rg=rot(np,2,.08)@rot(np,1,-.03)
 tg=np.array([.7,.2,.1]);tg/=np.linalg.norm(tg)
 K0=np.array([[800.,0,400],[0,800 if eq else 920.,300],[0,0,1.]])
 K1=np.array([[800. if eq else 1300.,0,310],[0,800. if eq else 1100.,240],[0,0,1.]])
 def proj(Y,K):
  h=Y@K.T
  return h[:,:2]/h[:,2,None]
 p0,p1=proj(X,K0),proj(X@Rg.T+tg,K1)
 p1+=np.array([[.15*math.sin(i),.12*math.cos(i)]for i in range(len(X))])
 return p0,p1,K0,K1,rot(np,0,.012)@Rg,tg+np.array([.01,-.02,.015]),np.ones(len(X),dtype=np.uint8)
def ref(np,p0,p1,K0,K1,R,t,z):
 t=t/np.linalg.norm(t)
 x,y,w=t
 E=np.array([[0,-w,y],[w,0,-x],[-y,x,0]])@R
 h0,h1=np.c_[p0,np.ones(len(p0))],np.c_[p1,np.ones(len(p1))]
 if z=="pixel":
  M,v0,v1,sc=np.linalg.inv(K1).T@E@np.linalg.inv(K0),h0,h1,1.
 else:
  M,v0,v1,sc=E,h0@np.linalg.inv(K0).T,h1@np.linalg.inv(K1).T,(K0[0,0]+K0[1,1]+K1[0,0]+K1[1,1])/4
 out=[]
 for a,b in zip(v0,v1):
  l1,l0=M@a,M.T@b
  out.append(sc*float(b@l1)/math.sqrt((sum(l1[:2]**2)+sum(l0[:2]**2))))
 return np.array(out)
@pt.mark.parametrize("z",["native","pixel"])
def test_fd(a,z):
 np,c=a;p0,p1,k0,k1,R,t,m=data(np)
 r,J,B=c.geometry(np,p0,p1,k0,k1,R,t,z)
 assert np.allclose(r,ref(np,p0,p1,k0,k1,R,t,z),rtol=1e-10,atol=1e-9)
 t=t/np.linalg.norm(t)
 assert np.allclose(B.T@B,np.eye(2),atol=1e-12)
 assert np.allclose(t@B,0,atol=1e-12)
 eps=1e-6
 for j in range(5):
  vals=[]
  for sign in[-1,1]:
   RR,tt=R.copy(),t.copy()
   if j<3:
    RR=rot(np,j,sign*eps)@R
   else:
    tt=t+sign*eps*B[:,j-3];tt/=np.linalg.norm(tt)
   vals.append(ref(np,p0,p1,k0,k1,RR,tt,z))
  fd=(vals[1]-vals[0])/(2*eps)
  assert np.allclose(J[:,j],fd,rtol=1e-8,atol=1e-6)
def test_eq(a):
 np,c=a
 for eq in[False,True]:
  p0,p1,k0,k1,R,t,m=data(np,eq)
  rp,Jp,_=c.geometry(np,p0,p1,k0,k1,R,t,"pixel")
  rn,Jn,_=c.geometry(np,p0,p1,k0,k1,R,t,"native")
  x0=np.c_[p0,np.ones(len(p0))]@np.linalg.inv(k0).T
  x1=np.c_[p1,np.ones(len(p1))]@np.linalg.inv(k1).T
  t=t/np.linalg.norm(t);x,y,z=t
  E=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])@R
  l1,l0=x0@E.T,x1@E
  dp=(l1[:,0]/k1[0,0])**2+(l1[:,1]/k1[1,1])**2+(l0[:,0]/k0[0,0])**2+(l0[:,1]/k0[1,1])**2
  assert np.allclose(rp,np.sum(x1*l1,axis=1)/np.sqrt(dp),atol=1e-9)
  if eq:
   assert np.allclose(rp,rn,atol=1e-9,rtol=1e-10)
   assert np.allclose(Jp,Jn,atol=1e-8,rtol=1e-10)
   a=c.polish(np,p0,p1,k0,k1,R,t,m,"pixel")
   b=c.polish(np,p0,p1,k0,k1,R,t,m,"native")
   assert np.allclose(a["R"],b["R"],atol=1e-8)
   assert np.allclose(a["t"],b["t"],atol=1e-8)
  else:
   assert np.max(np.abs(rp-rn))>.01
@pt.mark.parametrize("z",["native","pixel"])
def test_swap(a,z):
 np,c=a;p0,p1,k0,k1,R,t,m=data(np)
 r=c.geometry(np,p0,p1,k0,k1,R,t,z)[0]
 rr=c.geometry(np,p1,p0,k1,k0,R.T,-R.T@t,z)[0]
 assert np.allclose(r*r,rr*rr,atol=1e-8,rtol=1e-10)
def test_scale(a):
 np,c=a;p0,p1,k0,k1,R,t,m=data(np)
 h0,h1=np.c_[p0,np.ones(len(p0))],np.c_[p1,np.ones(len(p1))]
 x,y,z=t/np.linalg.norm(t)
 F=np.linalg.inv(k1).T@np.array([[0,-z,y],[z,0,-x],[-y,x,0]])@R@np.linalg.inv(k0)
 dF=np.arange(45,dtype=float).reshape(5,3,3)*1e-6
 r,J=c.residual_matrix(np,h0,h1,F,dM=dF)
 for sc in[.01,-7.,11.]:
  rr,JJ=c.residual_matrix(np,h0,h1,F*sc,dM=dF*sc)
  assert np.allclose(rr,r*math.copysign(1,sc),atol=1e-8,rtol=1e-10)
  assert np.allclose(JJ,J*math.copysign(1,sc),atol=1e-7,rtol=1e-10)
def test_ls(a):
 np,c=a
 J=np.array([[math.sin((i+1)*(j+2))for j in range(5)]for i in range(11)])
 r=np.array([math.cos(i*.3)for i in range(11)]);lam=.001
 reg=np.diag(np.sqrt(lam*np.maximum(np.sum(J*J,axis=0),1e-9)))
 want=np.linalg.lstsq(np.vstack((J,reg)),np.r_[-r,np.zeros(5)],rcond=None)[0]
 assert np.allclose(c.normal_step(np,r,J,lam),want,atol=1e-10,rtol=1e-9)
def test_caps(a):
 np,c=a;R=np.eye(3);t=np.array([1.,0,0])
 d=np.array([3.,4,0,6,8]);orig=d.copy()
 RR,tt,lim=c.retract(np,R,t,d)
 assert np.array_equal(d,orig)
 assert np.linalg.norm(lim[:3])==pt.approx(.1)
 assert np.linalg.norm(lim[3:])==pt.approx(.1)
 assert np.allclose(RR.T@RR,np.eye(3),atol=1e-12)
 assert np.linalg.det(RR)==pt.approx(1.)
 assert np.linalg.norm(tt)==pt.approx(1.)
@pt.mark.parametrize("z",["native","pixel"])
def test_path(a,z):
 np,c=a;args=data(np);p0,p1,k0,k1,R,t,m=args
 saved=[v.tobytes()for v in args]
 a=c.polish(np,*args[:-1],m,z)
 assert a["valid"]and a["attempts"]==3 and len(a["trace"])==3
 assert[v.tobytes()for v in args]==saved
 assert a["accepted_steps"]==sum(x["accepted"]for x in a["trace"])
 cost=a["initial_cost"]
 for i,row in enumerate(a["trace"]):
  assert row["attempt"]==i and row["cost_before"]==cost
  if row["accepted"]:
   assert row["cost_trial"]<cost
   cost=row["cost_trial"]
  assert row["lambda_after"]==(max(1e-12,row["lambda_before"]*.5)if row["accepted"]else min(1e12,row["lambda_before"]*10))
  assert np.linalg.norm(row["delta"][:3])<=.100000000001
  assert np.linalg.norm(row["delta"][3:])<=.100000000001
 assert cost==a["final_cost"]
 assert np.allclose(a["R"].T@a["R"],np.eye(3),atol=1e-10)
 assert np.linalg.norm(a["t"])==pt.approx(1.)
 assert not np.shares_memory(a["R"],R)
 m=m.copy();m[-3:]=0
 b=c.polish(np,p0,p1,k0,k1,R,t,m,z)
 other=p1.copy();other[-3:]+=12345
 d=c.polish(np,p0,other,k0,k1,R,t,m,z)
 assert np.array_equal(b["R"],d["R"])and np.array_equal(b["t"],d["t"])
 assert b["final_cost"]==d["final_cost"]
def test_zero(a):
 np,c=a;p0=np.array([[i*10.,(i%3)*8.]for i in range(8)])
 p1=p0+np.array([10.,0]);K=np.diag([1000.,1000.,1.])
 a=c.polish(np,p0,p1,K,K,np.eye(3),np.array([1.,0,0]),np.ones(8,dtype=bool),"pixel")
 assert a["attempts"]==3 and a["accepted_steps"]==0
 assert a["initial_cost"]==a["final_cost"]==0
 assert all(x["rejection"]=="no_decrease" for x in a["trace"])
@pt.mark.parametrize("error",[False,True])
def test_solve(a,error):
 np,c=a
 class LA:
  def __getattr__(self,k):return getattr(np.linalg,k)
  def solve(self,*a):
   if error:return np.full(5,np.nan)
   raise np.linalg.LinAlgError("fake")
 class P:
  linalg=LA()
  def __getattr__(self,k):return getattr(np,k)
 p0,p1,k0,k1,R,t,m=data(np)
 a=c.polish(P(),p0,p1,k0,k1,R,t,m,"native")
 assert a["valid"]and a["attempts"]==3 and a["accepted_steps"]==0
 assert all(x["rejection"]==("invalid_trial_geometry" if error else "singular_solve")and x["delta"]is None for x in a["trace"])
def test_bad(a):
 np,c=a;p0,p1,k0,k1,R,t,m=data(np)
 bad=p0.copy();bad[0,0]=float("nan")
 with pt.raises(ValueError):c.polish(np,bad,p1,k0,k1,R,t,m,"pixel",success=False)
 with pt.raises(ValueError):c.polish(np,p0,p1,k0,k1,R,t,m.astype(float),"pixel")
 with pt.raises(ValueError):c.geometry(np,p0[:,0],p1,k0,k1,R,t,"pixel")
 a=c.polish(np,p0,p1,k0,k1,np.eye(3),np.zeros(3),m,"pixel",success=False)
 assert not a["valid"]and a["reason"]=="failed_seed" and a["attempts"]==0 and a["R"]is None
 a=c.polish(np,p0,p1,k0,k1,R,np.zeros(3),m,"pixel")
 assert not a["valid"]and a["reason"]=="invalid_initial_geometry" and a["R"]is None
 for z in["native","pixel"]:
  x=np.zeros((6,2));k=np.eye(3)
  a=c.polish(np,x,x,k,k,np.eye(3),np.array([0.,0,1.]),np.ones(6,dtype=bool),z)
  assert not a["valid"]and a["attempts"]==0 and a["R"]is None
  with pt.raises(c.GeometryError):c.geometry(np,x,x,k,k,np.eye(3),np.array([0.,0,1.]),z)
def test_low(a):
 np,c=a;p0,p1,k0,k1,R,t,m=data(np)
 m[:]=0;m[:4]=1
 a=c.polish(np,p0,p1,k0,k1,R,t,m,"pixel")
 assert a["valid"]and a["selected_rows"]==4 and a["attempts"]==0
 assert a["initial_cost"]is None and a["final_cost"]is None
 assert np.array_equal(a["R"],R)
 assert np.allclose(a["t"],t/np.linalg.norm(t),atol=1e-15)
 r=c.geometry(np,p0,p1,k0,k1,a["R"],a["t"],"native",False)[0]
 want=ref(np,p0,p1,k0,k1,a["R"],a["t"],"native")
 assert np.array_equal(np.abs(r)<=1,np.abs(want)<=1)
