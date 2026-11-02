"""Identity-warp capture experiment; unchanged released XRefine and one fixed recrop.
Third-party model and preprocessing remain in their pinned licensed vendor files.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types

ROOT=Path('/Users/ducng/Desktop/workspace/SLAMResaerch')
NS='research/prof-capture-basin-20261019/'
BASE={'path':'research/ar-persistence-20261014/controller-v8.py','bytes':37614,
      'sha256':'3088d315e314645b573454afb96c63f62e397574fa7a2e78b53b0d7b1a8c631b'}
ARMS=('raw','one','sham','recrop','extrap','phase')


def enc(v):
    return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')


def pinned(p,cap=32*1048576):
    path=ROOT/p['path']; assert path.is_relative_to(ROOT) and '..' not in Path(p['path']).parts
    assert not path.is_symlink() and path.is_file() and path.stat().st_nlink==1
    b=path.read_bytes();assert len(b)==p['bytes']<=cap and hashlib.sha256(b).hexdigest()==p['sha256']
    return b


def load_source(p,name):
    m=types.ModuleType(name);m.__file__=str(ROOT/p['path']);sys.modules[name]=m
    exec(compile(pinned(p),m.__file__,'exec'),m.__dict__);return m


def run(request,deadline):
    admission=request['admission'];nonce=admission['nonce'];assert request['admission_sha256']==hashlib.sha256(enc(admission)).hexdigest()
    lock=json.loads(pinned(request['lock']));freeze=json.loads(pinned(request['freeze']))
    assert admission['bundle']==request['lock'] and admission['freeze']==request['freeze']
    assert admission['sources']==lock['sources'] and admission['protocol_sha256']==lock['protocol_sha256']
    assert admission['output_caps']==lock['output_caps']
    assert freeze['bundle']==request['lock'] and freeze['sources']==lock['sources']
    assert freeze['protocol_sha256']==lock['protocol_sha256']
    owner=json.loads((ROOT/'research/experiment-owner.lock/owner.json').read_bytes())
    assert owner['active_job']['nonce']==nonce and owner['active_job']['admission']==admission
    assert owner['active_job']['admission_sha256']==request['admission_sha256']
    outputs={};phase='BEFORE_SCORING';progress={'image_index':None,'completed_images':0,'stock_forwards':0,'rgb_extrema':[]}

    def write(name,b):
        assert time.monotonic()<deadline and name in admission['output_caps'] and name not in outputs
        assert len(b)<=admission['output_caps'][name]
        o=json.loads((ROOT/'research/experiment-owner.lock/owner.json').read_bytes())
        assert o['active_job']['nonce']==nonce and o['active_job']['admission']==admission
        path=ROOT/name;assert name.startswith(NS) and not path.exists() and not path.is_symlink()
        with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        assert path.read_bytes()==b
        p={'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};outputs[name]=p;return p

    try:
        c=load_source(BASE,'prof_capture_saved_controller')
        files=dict(freeze['dependencies']['files'])
        for p in freeze['inputs']+freeze['sources']:files[p['path']]=p
        gate=c.FrozenGate(files);sys.meta_path.insert(0,gate)
        for p in [c.SITE,ROOT/'benchmarks/xrefine',ROOT/'vendor/XRefine',ROOT/'vendor/XRefine/submodules/glue_factory']:
            sys.path.insert(0,str(p))
        # Scoped import compatibility: supply real OS architecture without the
        # stdlib processor helper spawning uname -p. Uniform across all arms.
        import platform
        os_architecture=os.uname().machine
        platform._Processor.get=staticmethod(lambda:os_architecture)
        def no_spawn(event,args):
            if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.system'}:
                raise RuntimeError('actor process spawning prohibited:'+event)
        sys.addaudithook(no_spawn)
        import numpy as np
        import torch
        import benchmark as bench
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        checker=load_source(lock['sources'][2],'prof_capture_independent_checker')
        fixtures=checker.fixtures(np)
        models,weights=bench.load_models(['xrefine_small_xfeat']);model=models['xrefine_small_xfeat']
        assert not model.training and model.patch_radius==5 and not model.adjust_only_second_keypoint
        assert model.net.spatial_argmax.temperature==1.0 and model.net.spatial_argmax.eps==1e-6
        assert all(p.device.type=='cpu' for p in model.parameters())
        checkpoint=ROOT/'vendor/XRefine/pretrained/xrefine_small_specific_xfeat.pth'
        expected=bench.load_checkpoint(checkpoint)['model']
        assert set(model.net.state_dict())==set(expected)
        assert all(torch.equal(v,expected[k]) for k,v in model.net.state_dict().items())
        del expected
        processor=bench.preprocessor();offsets=np.asarray(lock['protocol']['offsets'],dtype=np.float32)
        assert offsets.shape==(13,2)
        arrays={};hws=[];banks=[];times=[];sources=[];forward_count=0
        for i,item in enumerate(lock['protocol']['images']):
            progress['image_index']=i
            assert time.monotonic()<deadline
            b=pinned(item['image']);t=time.perf_counter()
            image,_=bench.prepare_image(ROOT/item['image']['path'],processor)
            preprocessing=time.perf_counter()-t
            assert image.device.type=='cpu' and image.dtype==torch.float32 and image.ndim==3 and image.shape[0]==3
            assert bool(torch.isfinite(image).all()), 'nonfinite author-processed RGB'
            minimum,maximum=float(image.min()),float(image.max())
            rgb={'minimum':minimum,'maximum':maximum,'validation_tolerance':2.**-20,
                 'tensor_sha256':hashlib.sha256(image.detach().numpy().tobytes(order='C')).hexdigest()}
            progress['rgb_extrema'].append(rgb)
            assert minimum>=-(2.**-20) and maximum<=1+(2.**-20), 'author-processed RGB exceeds fixed historical tolerance'
            h,w=map(int,image.shape[-2:]);hws.append([h,w])
            with np.load(io.BytesIO(pinned(item['features'])),allow_pickle=False) as z:kp=z['keypoints0'].copy()
            assert kp.shape==(2048,2) and kp.dtype==np.float32 and np.isfinite(kp).all();banks.append(kp)
            anchors_all=np.floor(kp)+np.float32(.5)
            valid=(anchors_all[:,0]>=24)&(anchors_all[:,0]<w-24)&(anchors_all[:,1]>=24)&(anchors_all[:,1]<h-24)
            ids=np.flatnonzero(valid)[:32].astype(np.int64);assert len(ids)==32
            anchors=anchors_all[ids];raw0=np.broadcast_to(anchors,(13,32,2)).copy()
            raw1=(raw0+offsets[:,None,:]).astype(np.float32)
            p0=torch.from_numpy(raw0.reshape(-1,2).copy());p1=torch.from_numpy(raw1.reshape(-1,2).copy())
            with torch.no_grad():
                t=time.perf_counter();one=model(p0,p1,image,image);first_time=time.perf_counter()-t;forward_count+=1;progress['stock_forwards']=forward_count
                t=time.perf_counter();sham=model(p0,p1,image,image);sham_time=time.perf_counter()-t;forward_count+=1;progress['stock_forwards']=forward_count
                t=time.perf_counter();recrop=model(one[0],one[1],image,image);recrop_time=time.perf_counter()-t;forward_count+=1;progress['stock_forwards']=forward_count
            one0,one1=[v.cpu().numpy().reshape(13,32,2).copy() for v in one]
            sham0,sham1=[v.cpu().numpy().reshape(13,32,2).copy() for v in sham]
            recrop0,recrop1=[v.cpu().numpy().reshape(13,32,2).copy() for v in recrop]
            t=time.perf_counter()
            extrap0=(raw0.astype(np.float64)+2*(one0.astype(np.float64)-raw0)).astype(np.float32)
            extrap1=(raw1.astype(np.float64)+2*(one1.astype(np.float64)-raw1)).astype(np.float32)
            extrap_time=time.perf_counter()-t
            t=time.perf_counter()
            phase0=(np.floor(one0.astype(np.float64))+.5+(one0.astype(np.float64)-raw0)).astype(np.float32)
            phase1=(np.floor(one1.astype(np.float64))+.5+(one1.astype(np.float64)-raw1)).astype(np.float32)
            phase_time=time.perf_counter()-t
            values=dict(ids=ids,anchors=anchors,offsets=offsets.copy(),raw0=raw0,raw1=raw1,one0=one0,one1=one1,
                        sham0=sham0,sham1=sham1,recrop0=recrop0,recrop1=recrop1,extrap0=extrap0,extrap1=extrap1,phase0=phase0,phase1=phase1)
            arrays.update({f'i{i}_{key}':value for key,value in values.items()})
            times.append(dict(image=i,preprocessing_seconds=preprocessing,first_seconds=first_time,
                              sham_extra_seconds=sham_time,recrop_extra_seconds=recrop_time,
                              recrop_total_seconds=first_time+recrop_time,sham_total_seconds=first_time+sham_time,
                              extrap_total_seconds=first_time+extrap_time,phase_total_seconds=first_time+phase_time,cases_per_forward=416))
            sources.append(dict(image=i,name=item['image']['path'],height=h,width=w,selected_ids=ids.tolist(),raw_sha256=hashlib.sha256(b).hexdigest(),processed_rgb=rgb))
            progress['completed_images']=i+1
        assert len(arrays)==120 and forward_count==24
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
        observation=write(NS+f'observations-{admission["attempt"]:02d}.npz',buffer.getvalue())
        # All predictions are durable and sealed before any task errors/gates below.
        phase='AFTER_SCORING'
        producer={}
        for radius,indices in [(0,[0]),(2,list(range(1,5))),(4,list(range(5,9))),(8,list(range(9,13)))]:
            table={}
            for arm in ARMS:
                all_errors=[]
                for i in range(8):
                    a=arrays[f'i{i}_{arm}0'][indices].astype(np.float64)
                    b=arrays[f'i{i}_{arm}1'][indices].astype(np.float64)
                    all_errors.extend(np.sqrt(np.sum((a-b)**2,axis=-1)).ravel().tolist())
                table[arm]=dict(count=len(all_errors),mean_error=sum(all_errors)/len(all_errors))
            producer[str(radius)]=table
        with np.load(io.BytesIO(pinned(observation)),allow_pickle=False) as z:sealed={k:z[k].copy() for k in z.files}
        audit=checker.validate(np,sealed,hws,banks)
        assert audit['status']=='PASS' and audit['complete'] is True
        for radius,table in producer.items():
            for arm,row in table.items():
                assert abs(row['mean_error']-audit['radius_metrics'][radius][arm]['mean_error'])<1e-10
        report=dict(schema='prof-capture-report-v1',question=lock['question'],nonce=nonce,
                    protocol_sha256=lock['protocol_sha256'],complete=True,scoring_started=True,
                    outcome='DIAGNOSTIC_POSITIVE' if audit['advance'] else 'SCIENTIFIC_NEGATIVE',
                    observations=observation,known_relation='identity warp; same processed RGB on both sides',
                    all_cases=3328,primary_cases=2048,forward_count=forward_count,images=sources,
                    unique_images=len({s['raw_sha256'] for s in sources}),timings=times,fixtures=fixtures,
                    producer_radius_metrics=producer,audit=audit,weights=weights,
                    import_compatibility={'platform_processor_scope':'real OS architecture; no subprocess','os_architecture':os_architecture},
                    limits=['development identity diagnostic','not real cross-view pose accuracy','no novelty claim','fixed extra pass; no confidence-gated speed saving','conditional cold ordered timings at416 cases; no full-pipeline benchmark'])
        write(NS+f'result-{admission["attempt"]:02d}.json',enc(report))
        return dict(schema='prof-capture-actor-result-v1',status='PASS',complete=True,nonce=nonce,outputs=outputs)
    except BaseException as exc:
        failure=dict(schema='prof-capture-failure-v1',nonce=nonce,scoring_started=phase!='BEFORE_SCORING',phase=phase,
                     error=type(exc).__name__+':'+str(exc)[:4000],progress=progress,traceback=traceback.format_exc()[-16000:])
        path=ROOT/(NS+f'pre-scoring-failure-{admission["attempt"]:02d}.json')
        failure_body=enc(failure)
        assert path.relative_to(ROOT).as_posix()==admission['failure_path']
        assert len(failure_body)<=admission['failure_cap_bytes']
        current=json.loads((ROOT/'research/experiment-owner.lock/owner.json').read_bytes())
        assert current['active_job']['nonce']==nonce and current['active_job']['admission']==admission
        if not path.exists():
            with path.open('xb') as f:f.write(failure_body);f.flush();os.fsync(f.fileno())
            assert path.read_bytes()==failure_body
        raise


if __name__=='__main__':
    assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize
    fd=int(sys.argv[1]);deadline=float(sys.argv[2]);signal.signal(signal.SIGALRM,signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    raw=sys.stdin.buffer.read(65537);assert len(raw)<=65536
    result=run(json.loads(raw),deadline);body=enc(result)
    assert len(body)<=131072
    with os.fdopen(fd,'wb') as f:f.write(body);f.flush()
