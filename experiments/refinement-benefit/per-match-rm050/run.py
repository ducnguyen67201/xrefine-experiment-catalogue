"""RM050 bounded weak-risk screen, reusing authenticated RM045 cache/runtime seams.
No images, native fitting, MD or held FINAL truth. One serialized local worker.
"""
from pathlib import Path
import datetime as dt
import fcntl, hashlib, importlib.util, io, json, os, re, resource, signal, stat
import subprocess, sys, time, traceback
HERE=Path(__file__).absolute().parent
WORK=HERE.parents[1]
ROOT=WORK/'research-execution/autonomous-xrefine-20261102/rm050-risk-screen/attempt-001'
GIB=1073741824
PHASES=('checks','fit-predict','score')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def enc(obj):return (json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def pin(path):
    path=Path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        a=os.fstat(fd);need(stat.S_ISREG(a.st_mode) and a.st_nlink==1,'regular single-link input');h=hashlib.sha256()
        while True:
            b=os.read(fd,262144)
            if not b:break
            h.update(b)
        b=os.fstat(fd);stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
        need(stamp(a)==stamp(b)==stamp(path.lstat()),'input stable')
        return dict(path=str(path.absolute()),bytes=a.st_size,sha256=h.hexdigest())
    finally:os.close(fd)

def read(p):
    need(pin(p['path'])==p,'exact pin '+p['path']);return Path(p['path']).read_bytes()

def write(path,obj):
    body=obj if isinstance(obj,bytes) else enc(obj)
    need(len(body)<=64*1048576,'bounded stage artifact');fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(body);f.flush();os.fsync(f.fileno())
    need(Path(path).read_bytes()==body,'full write readback');return pin(path)

def load(p,name):
    b=read(p);spec=importlib.util.spec_from_file_location(name,p['path']);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;exec(compile(b,p['path'],'exec'),m.__dict__);need(pin(p['path'])==p,'loaded source stable');return m

def config():
    c=json.loads((HERE/'request.json').read_bytes())
    for p in c['sources'].values():read(p)
    cp=pin(HERE/'request.json');checkpoint=json.loads((ROOT/'checkpoint.json').read_bytes())
    need(checkpoint['request']==cp and checkpoint['sources']==c['sources'],'preverification source/config checkpoint')
    return c,cp

def npz(route,path,arrays):
    import numpy as np
    b=io.BytesIO();np.savez(b,**arrays);return write(path,b.getvalue())

def barrier(c,route,folder):
    import numpy as np
    lead=load(c['sources']['base_core'],'_rm050_frozen_base_cache_contract')
    saved=json.loads(read(c['feature_seal']));records=route.validate_membership(json.loads(route.read_pin(route.MEMBERSHIP,17*1048576)))
    need(saved['membership']==route.MEMBERSHIP and saved['records_sha256']==route.digest(records),'same original300 membership/seal')
    result=route.seal_all300(records,saved['receipts'],folder/'all300-feature-seal.json',lead)
    write(folder/'barrier-event.json',dict(status='ALL300_AUTHENTICATED_BEFORE_SELECTED_TRUTH',feature_seal=result.seal_pin,fit=list(range(120)),hold=list(range(120,150)),DEV_or_FINAL_truth=False))
    return result

def targets(route,b,indices,folder):
    """Minimal selected-body adaptation: no eager150-target/opaque pair-GT loader."""
    import numpy as np
    need(indices==list(range(120)) or indices==list(range(120,150)),'fixed TRAIN120 or HOLD30 only')
    # ALL300 actual array and source seals were completed by barrier() first.
    need((folder/'barrier-event.json').is_file(),'truth follows actual saved ALL300 barrier')
    inventory={}
    for line in route.read_pin(route.INVENTORY,4*1048576).splitlines():
        row=json.loads(line);need(row['member'] not in inventory,'unique original inventory');inventory[row['member']]=row
    need(len(inventory)==8190,'complete original inventory')
    result={};opened=[]
    for index in indices:
        r=b.records[index];actor=r['original_actor_record'];room=707+index//15
        names=[re.fullmatch(r'(scene[0-9]{4}_[0-9]{2})/color/([0-9]+)\.(?:jpg|jpeg|png)',name) for name in actor['image_names']]
        need(all(names) and names[0][1]==names[1][1] and type(actor['space_id']) is int and actor['space_id']==room and names[0][1].startswith(f'scene{room:04d}_'),'selected original room/scan/direction')
        need(all(type(actor['scan_ids'][side]) is int and int(name[1].rsplit('_',1)[1])==actor['scan_ids'][side] for side,name in enumerate(names)),'exact original endpoint scan frame IDs')
        cameras=[]
        for side,name in enumerate(names):
            member=f'scannet1500/{name[1]}/pose/{name[2]}.txt';row=inventory[member]
            need(row['classification']=='private_authority' and row['path']==str(route.SCN_STAGE/'scannet-extracted/private-authority'/member),'original selected private pose locator')
            p={key:row[key] for key in ('path','bytes','sha256')};body=route.read_pin(p,16384)
            values=[float(x) for x in body.decode('ascii').split()];need(len(values)==16,'original16 C2W')
            C=np.asarray(values,np.float64).reshape(4,4)
            need(np.isfinite(C).all() and np.array_equal(C[3],[0.,0.,0.,1.]) and np.allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=1e-5,rtol=0) and np.isclose(np.linalg.det(C[:3,:3]),1.,atol=1e-5,rtol=0),'proper original C2W no repair')
            cameras.append(C);opened.append(dict(index=index,side=side,pin=p))
        T=np.linalg.inv(cameras[1])@cameras[0]
        result[index]=dict(K0=np.asarray(r['frames'][0]['K'],np.float64),K1=np.asarray(r['frames'][1]['K'],np.float64),R=T[:3,:3].copy(),t=T[:3,3].copy())
    write(folder/'selected-truth-reads.json',dict(indices=indices,opened=opened,no_other_pose_body=True,barrier=pin(folder/'barrier-event.json')))
    return result

def cache(route,b,index):
    return route.npz_arrays(b.receipts[index]['cache_pin'],route.CACHE_ARRAYS)

def fit_predict(c,route,math,folder):
    import numpy as np
    b=barrier(c,route,folder);truth=targets(route,b,list(range(120)),folder)
    moments=math.FeatureMoments(1156);smallmoments=math.FeatureMoments(4);labels=[];counts=[]
    for i in range(120):
        a=cache(route,b,i);X=math.features(a);D=math.displacement(a);moments.add(X);smallmoments.add(D)
        labels.append(math.action_risks(a,math.fundamental(**truth[i]))[:,:4]);counts.append(len(X))
    mean,std=moments.finish();dm,ds=smallmoments.finish();acc=math.RidgeAccumulator(mean,std);shuffle=math.RidgeAccumulator(mean,std);disp=math.RidgeAccumulator(dm,ds)
    Y=np.concatenate(labels);mapping=np.arange(len(Y),dtype=np.int64);rng=np.random.default_rng(20261102);offset=0
    for room in range(8):
        n=sum(counts[room*15:(room+1)*15]);mapping[offset:offset+n]=offset+rng.permutation(n);offset+=n
    need(offset==len(Y),'whole jointvector shuffle withinTRAINroom');offset=0
    for i in range(120):
        a=cache(route,b,i);X=math.features(a);n=len(X);acc.add(X,Y[offset:offset+n]);shuffle.add(X,Y[mapping[offset:offset+n]]);disp.add(math.displacement(a),Y[offset:offset+n]);offset+=n
    W=acc.solve();S=shuffle.solve();D=disp.solve()
    weights=npz(route,folder/'ridge-models.npz',dict(mean=mean,std=std,W=W,S=S,displacement_mean=dm,displacement_std=ds,D=D,shuffle_mapping=mapping))
    frozen=write(folder/'model-freeze.json',dict(status='FROZEN_BEFORE_ANY_HOLD_TRUTH',weights=weights,training_indices=list(range(120)),training_rows=len(Y),training_label_sha256=hashlib.sha256(Y.tobytes()).hexdigest(),seed=20261102,actions=['QQ','PP','PQ','QP'],directional_features=True,ridge='SUM_SQUARED_LOSS_PLUS_ALPHA1_WEIGHTS_UNPENALIZED_INTERCEPT',request=pin(HERE/'request.json'),feature_seal=b.seal_pin))
    predictions=[]
    for i in range(120,150):
        a=cache(route,b,i);X=math.features(a);C=math.predict(X,mean,std,W);SC=math.predict(X,mean,std,S);DC=math.predict(math.displacement(a),dm,ds,D)
        p=npz(route,folder/f'prediction-{i:04d}.npz',dict(row_ids=a['row_ids'],candidate_scores=C,shuffle_scores=SC,displacement_scores=DC,candidate=math.choose(C),shuffled=math.choose(SC),displacement=math.choose(DC)))
        predictions.append(dict(index=i,uid=b.receipts[i]['uid'],pin=p,feature=b.receipts[i]['cache_pin']))
    seal=write(folder/'all30-predictions-seal.json',dict(status='ALL30_PREDICTIONS_BEFORE_HOLD_TRUTH',model=frozen,predictions=predictions,held_truth_bodies_opened=0,held_risks_used_for_fit_or_action=False))
    return dict(model=frozen,predictions=seal,training_rows=len(Y),prediction_pairs=30,barrier=b.seal_pin)

def score(c,route,math,folder):
    import numpy as np
    previous_terminal=json.loads((ROOT/'fit-predict/terminal.json').read_bytes());need(previous_terminal['status']=='passed' and previous_terminal['closure']['complete'] and previous_terminal['exit_code']==0,'actual fit worker terminal/closure')
    prior_receipt=json.loads(read(previous_terminal['receipt']));checkpoint=json.loads((ROOT/'checkpoint.json').read_bytes());request_pin=pin(HERE/'request.json')
    need(prior_receipt['status']=='passed' and prior_receipt['phase']=='fit-predict' and prior_receipt['request']==request_pin and prior_receipt['source_checkpoint']==checkpoint['commit'],'fit receipt exact source/config/phase')
    previous=prior_receipt['result'];seal=json.loads(read(previous['predictions']));model=json.loads(read(seal['model']));read(model['weights']);read(model['feature_seal'])
    need(seal['model']==previous['model'] and model['request']==request_pin and model['feature_seal']==previous['barrier'],'model linked to frozen request and actual fit barrier')
    need(seal['held_truth_bodies_opened']==0 and model['training_indices']==list(range(120)) and [r['index'] for r in seal['predictions']]==list(range(120,150)),'all30 and model frozen priortruth')
    for r in seal['predictions']:read(r['pin'])
    write(folder/'prediction-barrier.json',dict(predictions=previous['predictions'],model=seal['model'],status='ALL_HELD_ACTIONS_REAUTHENTICATED_BEFORE_HOLD_TRUTH'))
    b=barrier(c,route,folder)
    need(model['feature_seal']['sha256']==b.seal_pin['sha256'],'identical complete feature seal content in fit and score')
    for r in seal['predictions']:need(r['feature']==b.receipts[r['index']]['cache_pin'] and r['uid']==b.receipts[r['index']]['uid'],'all held prediction/source/UID joins before truth')
    truth=targets(route,b,list(range(120,150)),folder)
    methods=('candidate','displacement','shuffled','always_QQ','always_PP','fixed_half');pairs=[];rowfiles=[]
    for item in seal['predictions']:
        i=item['index'];a=cache(route,b,i);saved=route.npz_arrays(item['pin'],('row_ids','candidate_scores','shuffle_scores','displacement_scores','candidate','shuffled','displacement'))
        need(np.array_equal(saved['row_ids'],a['row_ids']),'exact held row IDs')
        for method,field in [('candidate','candidate_scores'),('shuffled','shuffle_scores'),('displacement','displacement_scores')]:need(np.array_equal(math.choose(saved[field]),saved[method]),'sealed argmin action lineage')
        losses=math.action_risks(a,math.fundamental(**truth[i]));n=len(losses);oracle=losses[:,:4].min(axis=1);decisions={m:saved[m] for m in methods[:3]};decisions.update(always_QQ=np.zeros(n,dtype=np.int64),always_PP=np.ones(n,dtype=np.int64),fixed_half=np.full(n,4,dtype=np.int64))
        rows={'row_ids':a['row_ids'],'action_risks':losses}
        summary=dict(index=i,room=707+i//15,rows=n,oracle_mean=float(oracle.mean()),methods={})
        for m,ch in decisions.items():
            chosen=losses[np.arange(n),ch];rows[m]=ch
            summary['methods'][m]=dict(weak_risk=float(chosen.mean()),regret_vs_four_action_oracle=float((chosen-oracle).mean()),actions=np.bincount(ch,minlength=5).tolist())
        rowfiles.append(npz(route,folder/f'weak-rows-{i:04d}.npz',rows));pairs.append(summary)
    byroom={}
    for room in (715,716):byroom[str(room)]={m:float(sum(p['rows']*p['methods'][m]['weak_risk'] for p in pairs if p['room']==room)/sum(p['rows'] for p in pairs if p['room']==room)) for m in methods}
    macro={m:float(np.mean([byroom[str(room)][m] for room in (715,716)])) for m in methods}
    gates={m:dict(macro_one_percent=macro['candidate']<=.99*macro[m],room715=byroom['715']['candidate']<byroom['715'][m],room716=byroom['716']['candidate']<byroom['716'][m]) for m in methods[1:]}
    passed=all(all(g.values()) for g in gates.values())
    result=dict(status='QUALIFIED_FOR_POSE_STAGE' if passed else 'VALID_NEGATIVE_CLOSED',weak_risk_only=True,pose_benefit_established=False,physical_localization_established=False,methods=methods,rows=sum(p['rows'] for p in pairs),pairs=pairs,room_pooled_row_risk=byroom,room_macro_risk=macro,gates=gates,all_gates_pass=passed,prediction_seal=previous['predictions'],rowfiles=rowfiles,truth_reads=pin(folder/'selected-truth-reads.json'),model=seal['model'],counter=dict(guard=0,crash=0))
    p=write(folder/'screen-result.json',result)
    return dict(result=p,status=result['status'],macro=macro,gates=gates)

def child(phase):
    c,cp=config();route=load(c['sources']['route'],'_rm050_original_route');runtime=route.bootstrap_runtime('fit');math=load(c['sources']['math'],'math050');folder=ROOT/phase
    if phase=='checks':
        import pytest
        code=pytest.main([c['sources']['tests']['path'],'-q','-p','no:cacheprovider']);need(code==0,'meaningful affected math tests');result=dict(pytest_exit=int(code),test_source=c['sources']['tests'])
    elif phase=='fit-predict':result=fit_predict(c,route,math,folder)
    elif phase=='score':result=score(c,route,math,folder)
    else:raise ValueError('phase')
    write(folder/'receipt.json',dict(status='passed',phase=phase,result=result,request=cp,runtime=runtime,source_checkpoint=json.loads((ROOT/'checkpoint.json').read_bytes())['commit']))
    print(json.dumps(dict(status='passed',phase=phase,result=result)),flush=True)

def retained():
    size=files=0
    for p in ROOT.rglob('*'):
        st=p.lstat();need(not stat.S_ISLNK(st.st_mode),'owned no symlink outputs')
        if stat.S_ISREG(st.st_mode):need(st.st_nlink==1,'owned singlelink outputs');size+=st.st_size;files+=1
    need(size<=256*1048576,'actual stage output forecast256MiB');return dict(bytes=size,files=files)

def parent():
    c,cp=config();local=load(c['sources']['supervisor'],'_rm050_existing_supervisor');lock=os.open(local.LOCK,os.O_RDWR|os.O_NOFOLLOW);st=os.fstat(lock)
    need((st.st_dev,st.st_ino,stat.S_IMODE(st.st_mode),st.st_nlink)==(16777231,204740392,0o600,1),'shared stable lock identity')
    stages=[];owner=local.census()[os.getpid()];started=time.monotonic()
    try:
        fcntl.flock(lock,fcntl.LOCK_EX)
        capacity=local.capacity();need(capacity['disk_available']>=capacity['disk_required']+8*GIB,'old liabilities plus prospective8GiBreservation plus32GiBfloor')
        write(ROOT/'launch.json',dict(owner=owner,request=cp,capacity=capacity,new_campaign_reservation_bytes=8*GIB,stage_forecast_bytes=256*1048576,old_limits_unchanged=True,prior_closure=c['preceding_closure']))
        prior=json.loads(read(c['preceding_closure']));need(prior['parent_and_children_absent'] and prior['owned_groups_empty'] and prior['worker_FDs_closed'] and prior['lockFD_EBADF'],'previous diagnosis positiveclosure')
        for phase in PHASES:
            folder=ROOT/phase;folder.mkdir();known={};samples=[];error=None;signals=[];process=None;begin=time.monotonic()
            argv=['/usr/bin/time','-p',local.PYTHON,'-I','-S','-B',str(Path(__file__).absolute()),'--child',phase]
            command=write(folder/'command.json',dict(argv=argv,request=cp,cpu=4,inclusive_RSS=8*GIB,wall_seconds=1800,TERM5_KILL5=True))
            with (folder/'stdout.txt').open('xb') as out,(folder/'stderr.txt').open('xb') as err:
                try:
                    process=subprocess.Popen(argv,cwd=WORK,env=local.ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,start_new_session=True)
                    write(folder/'started.json',dict(pid=process.pid,pgid=process.pid,owner=owner))
                    while process.poll() is None:
                        current=local.family(local.census(),process,known);rss=sum(r['rss'] for r in current)+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;samples.append(dict(seconds=time.monotonic()-begin,inclusive_RSS=rss,owned=current));need(rss<=8*GIB,'inclusiveRSS8GiB');need(time.monotonic()-begin<1800,'stagewall1800s');retained();time.sleep(1)
                except BaseException:error=traceback.format_exc()
                finally:
                    if process is not None:
                        for sig in (signal.SIGTERM,signal.SIGKILL):
                            return_code=process.poll();current=local.family(local.census(),process,known)
                            if not current and return_code is not None:break
                            for row in current:
                                try:os.kill(row['pid'],sig);signals.append(dict(pid=row['pid'],birth=row['birth'],signal=int(sig)))
                                except ProcessLookupError:pass
                            deadline=time.monotonic()+5
                            while time.monotonic()<deadline:
                                return_code=process.poll();current=local.family(local.census(),process,known)
                                if not current and return_code is not None:break
                                time.sleep(.1)
                        return_code=process.poll();unresolved=local.family(local.census(),process,known)
                        if unresolved or return_code is None:
                            write(folder/'unresolved-closure.json',dict(status='RESERVATION_AND_LOCK_RETAINED',alive=unresolved,known_generations=known))
                            # No new job/edit/lock release while owned processes remain.
                            while True:
                                return_code=process.poll();unresolved=local.family(local.census(),process,known)
                                if not unresolved and return_code is not None:break
                                time.sleep(5)
                        process.wait(timeout=5)
                    out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
            alive=local.family(local.census(),process,known) if process else [];complete=not alive and (process is None or process.returncode is not None)
            status='passed' if error is None and process is not None and process.returncode==0 and complete and (folder/'receipt.json').exists() else 'failed'
            terminal=write(folder/'terminal.json',dict(status=status,owner=owner,phase=phase,command=command,enclosing_seconds=time.monotonic()-begin,exit_code=None if process is None else process.returncode,error=error,closure=dict(complete=complete,known_generations=known,alive=alive,child_group=None if process is None else process.pid,stdout_stderr_FDs_closed=True,signals=signals),samples=samples,stdout=pin(folder/'stdout.txt'),stderr=pin(folder/'stderr.txt'),receipt=pin(folder/'receipt.json') if (folder/'receipt.json').exists() else None,retained=retained()))
            stages.append(terminal);print(json.dumps(dict(phase=phase,status=status,terminal=terminal)),flush=True);need(status=='passed','stage failed; preserve and classify earliest actual failure')
    finally:
        os.close(lock)
        try:os.fstat(lock);raise ValueError('live closedlockFD')
        except OSError as e:need(e.errno==9,'lockFD EBADF')
        write(ROOT/'parent-closure.json',dict(owner=owner,stages=stages,lockFD_EBADF=True,elapsed_enclosing_seconds=time.monotonic()-started,nested_stage_time_not_added=True,source_review_audit_publication_overhead='NONZERO_UNKNOWN'))

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--child':child(sys.argv[2])
    elif len(sys.argv)==1:parent()
    else:raise ValueError('unassigned argv')
