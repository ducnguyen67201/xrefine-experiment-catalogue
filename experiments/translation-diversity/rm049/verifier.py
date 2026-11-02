"""RM049 producer: frozen translation-normal diversity, no pose or GT imports."""
import math
ARMS = ['star_small','quality_coverage','confidence_only','equal_random',
        'quality_within_shuffled','quality_joint_shuffled','aligned_topk',
        'quality_score_only','star_native','stock_small']

def need(ok, text):
    if not ok: raise ValueError(text)

def target_count(n):
    return min(n,max(5,(n+1)//2))

def rank(np, values, k):
    need(values.ndim==1 and np.isfinite(values).all() and 0<=k<=len(values),'finite rank')
    return np.sort(np.lexsort((np.arange(len(values)),-values))[:k]).astype(np.int64)

def eligible(np, score, conf, cuts):
    need(score.shape==conf.shape and score.ndim==1 and np.isfinite(score).all()
         and np.isfinite(conf).all() and all(math.isfinite(cuts[t]) for t in ('margin','confidence')),'finite gate')
    return (score>=cuts['margin']) & (conf>=cuts['confidence'])

def policy(np, p0,p1,hw0,hw1,k):
    n=len(p0)
    need(p0.shape==p1.shape==(n,2) and np.isfinite(p0).all() and np.isfinite(p1).all(),'finite raw endpoints')
    need(0<=k<=n and (not n or k>0),'eligible coverage count')
    raw=np.column_stack((p0/np.asarray(hw0[::-1]),p1/np.asarray(hw1[::-1]))).astype(np.float64)
    outside=((raw<0)|(raw>=1)).any(axis=1)
    for g in (4,2,1):
        labels=np.clip(np.floor(raw*g),0,g-1).astype(np.int64)
        cells=((labels[:,0]*g+labels[:,1])*g+labels[:,2])*g+labels[:,3]
        occupied,counts=np.unique(cells,return_counts=True)
        if len(occupied)<=k: break
    c=len(occupied);q=np.ones(c,dtype=np.int64);R,D=k-c,n-c
    if D:
        dm=[divmod(R*(int(x)-1),D) for x in counts]
        q+=np.asarray([x[0] for x in dm],dtype=np.int64)
        order=sorted(range(c),key=lambda i:(-dm[i][1],int(occupied[i])))
        for i in order[:k-int(q.sum())]:q[i]+=1
    need(int(q.sum())==k and bool(np.all(q<=counts)),'exact eligible capacities')
    return dict(grid=np.asarray(g,dtype=np.int64),cells=cells,labels=labels,
                occupied=occupied,quotas=q,outside=outside)

def select_cells(np,values,p):
    chosen=[]
    for c,q in zip(p['occupied'],p['quotas']):
        rows=np.flatnonzero(p['cells']==c)
        chosen.extend(rows[np.lexsort((rows,-values[rows]))[:int(q)]].tolist())
    return np.asarray(sorted(chosen),dtype=np.int64)

ARMS=["star_small","translation_diversity","eligible_confidence","eligible_margin","eligible_random","geometry_shuffled","global_confidence","global_aligned","star_native","stock_small"]
EPS=1e-12
PSD=1e-12
RTOL=1e-5
CUTS=dict(margin=0.02997952699661255,confidence=0.2813373804092407)

def spectrum(np,matrices):
    need(matrices.dtype==np.float64 and matrices.ndim==3 and matrices.shape[1:]==(3,3) and np.isfinite(matrices).all(),'finite float64 matrix batch')
    need(bool(np.all(np.max(np.abs(matrices-matrices.transpose(0,2,1)),axis=(1,2))<=PSD)),'symmetric matrices')
    raw=np.linalg.eigvalsh((matrices+matrices.transpose(0,2,1))*.5)
    need(np.isfinite(raw).all() and bool(np.all(raw[:,0]>=-PSD)),'PSD absolute tolerance')
    ev=np.maximum(raw,0.);trace=ev.sum(axis=1);gap=ev[:,1]-ev[:,0]
    return raw,ev,trace,gap,gap/(trace+EPS),ev[:,0]/(trace+EPS)

def information(np,matrix,count):
    raw,ev,tr,gap,q,plane=spectrum(np,np.asarray(matrix,dtype=np.float64)[None])
    need(count>0 or float(tr[0])<=PSD,'zero-count information')
    return dict(matrix=(matrix+matrix.T)*.5,raw_eigenvalues=raw[0],eigenvalues=ev[0],count=int(count),trace=float(tr[0]),Q=float(q[0]),absolute_gap=float(gap[0]),absolute_gap_per_row=float(gap[0])/count if count else 0.,coplanarity=float(plane[0]),clipped_negative_eigenvalues=int((raw[0]<0).sum()))

def essential(np,R,t):
    length=float(np.linalg.norm(t));need(length>EPS and math.isfinite(length),'successful nonzero t')
    x,y,z=t/length
    return np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=np.float64)@R

def model_metrics(np,models,support=None):
    need(len(models)==3,'all/even/odd models')
    success=[bool(m['success']) for m in models]
    for m,ok in zip(models,success):
        R,t=m['R'],m['t']
        need(R.dtype==t.dtype==np.float64 and R.shape==(3,3) and t.shape==(3,) and np.isfinite(R).all() and np.isfinite(t).all(),'saved finite model')
        if ok:
            need(float(np.max(np.abs(R.T@R-np.eye(3))))<=RTOL and abs(float(np.linalg.det(R))-1)<=RTOL,'proper successful R')
            need(float(np.linalg.norm(t))>EPS,'successful nonzero t')
    if not all(success):return dict(available=False,successes=success,max_rotation_spread_deg=None,max_unsigned_translation_spread_deg=None,min_heldout_support=None,heldout_support={'even_to_odd':None,'odd_to_even':None})
    need(support is not None,'opposite support required')
    ar=[];at=[]
    for i in range(3):
        for j in range(i+1,3):
            v=(float(np.trace(models[i]['R']@models[j]['R'].T))-1)/2
            ar.append(math.degrees(math.acos(min(1.,max(-1.,v)))))
            a,b=models[i]['t'],models[j]['t'];v=abs(float(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))))
            at.append(math.degrees(math.acos(min(1.,max(0.,v)))))
    return dict(available=True,successes=success,max_rotation_spread_deg=max(ar),max_unsigned_translation_spread_deg=max(at),min_heldout_support=min(support.values()),heldout_support=support)

def geometry(np,bank,capture,record,models,cap):
    n=len(bank['primary0']);ids=bank['acquired_row_ids'];need(np.array_equal(ids,np.arange(n,dtype=np.int64)),'actual original row IDs')
    need(all(bool(m['success']) for m in models),'geometry only after all successful prepasses')
    q0,q1=bank['primary0'],bank['primary1'];K0=np.asarray(record['K0'],dtype=np.float64);K1=np.asarray(record['K1'],dtype=np.float64)
    x0=np.column_stack((q0.astype(np.float64),np.ones(n)));x1=np.column_stack((q1.astype(np.float64),np.ones(n)))
    i0,i1=np.linalg.inv(K0),np.linalg.inv(K1);h0=x0@i0.T;h1=x1@i1.T
    need(np.isfinite(h0).all() and np.isfinite(h1).all() and bool(np.all(h0[:,2]>EPS)) and bool(np.all(h1[:,2]>EPS)),'forward finite calibrated bearings')
    f0=h0/np.linalg.norm(h0,axis=1)[:,None];f1=h1/np.linalg.norm(h1,axis=1)[:,None]
    residuals=[]
    for m,parity in zip(models[1:],(0,1)):
        need(np.array_equal(m['fit_ids'],ids[ids%2==parity]),'actual complementary fit IDs')
        F=i1.T@m['E']@i0;a=x0@F.T;b=x1@F
        den=np.sqrt(np.sum(a[:,:2]**2+b[:,:2]**2,axis=1));need(np.isfinite(den).all() and bool(np.all(den>EPS)),'finite nondegenerate Sampson denominator')
        residuals.append(np.abs(np.sum(x1*a,axis=1))/den)
    residual=np.where(ids%2==0,residuals[1],residuals[0]);need(np.isfinite(residual).all(),'finite opposite-parity residual')
    normal=np.cross(f0@models[0]['R'].T,f1);norm=np.linalg.norm(normal,axis=1)
    capped=normal*np.minimum(1.,cap/np.maximum(norm,EPS))[:,None]
    conf=capture['v_confidence'].astype(np.float64);need(np.isfinite(conf).all() and bool(np.all((conf>=0)&(conf<=1))),'confidence in [0,1]')
    robust=np.empty(n,dtype=np.float64);small=residual<=1
    robust[small]=1/(1+residual[small]**2);inv=1/residual[~small];robust[~small]=inv**2/(1+inv**2)
    weight=np.clip(conf,0.,1.)*robust;A=weight[:,None,None]*capped[:,:,None]*capped[:,None,:]
    support={'even_to_odd':float(np.mean(residual[ids%2==1]<=1.)),'odd_to_even':float(np.mean(residual[ids%2==0]<=1.))}
    return dict(normals=normal,normal_norms=norm,capped_normals=capped,opposite_parity_residual=residual,weights=weight,contributions=A,metrics=model_metrics(np,models,support))

def calibrate(records):
    need(len(records)==3 and [r['record_id'] for r in records]==[0,1,2] and all(r['room_id']==707 for r in records),'exact all-three offpilot records')
    for r in records:
        need(r['metrics']['available'] and all(r['metrics']['successes']),'all calibration prepasses successful')
        need(r['metrics']['min_heldout_support']>0 and r['information']['absolute_gap_per_row']>EPS and r['information']['trace']>EPS,'positive offpilot support and information premise')
    return dict(max_rotation_spread_deg=max(r['metrics']['max_rotation_spread_deg'] for r in records),max_unsigned_translation_spread_deg=max(r['metrics']['max_unsigned_translation_spread_deg'] for r in records),min_heldout_support=min(r['metrics']['min_heldout_support'] for r in records),min_absolute_gap_per_row=min(r['information']['absolute_gap_per_row'] for r in records),max_coplanarity=max(r['information']['coplanarity'] for r in records))

def reliable(metrics,info,bounds):
    reasons=[]
    if not metrics['available']:reasons.append('prepass_failure')
    else:
        for key,why in [('max_rotation_spread_deg','rotation_inconsistent'),('max_unsigned_translation_spread_deg','translation_inconsistent')]:
            if metrics[key]>bounds[key]:reasons.append(why)
        if metrics['min_heldout_support']<bounds['min_heldout_support']:reasons.append('fit_excluded_support_low')
    if not info['count'] or info['absolute_gap_per_row']<bounds['min_absolute_gap_per_row']:reasons.append('absolute_information_low')
    if info['coplanarity']>bounds['max_coplanarity']:reasons.append('coplanarity_high')
    return dict(accepted=not reasons,reasons=reasons,information_population='WHOLE TRUE ELIGIBLE, before selection')

def greedy(np,A,ids,pol,confidence,k):
    count=len(ids);used=np.zeros(len(pol['occupied']),dtype=np.int64);cellpos=np.searchsorted(pol['occupied'],pol['cells']);done=np.zeros(count,dtype=bool)
    matrix=np.zeros((3,3),dtype=np.float64);order=[];history=[];evaluations=0
    while len(order)<k:
        feasible=np.flatnonzero((~done)&(used[cellpos]<pol['quotas'][cellpos]));need(len(feasible)>0,'feasible exact quota completion')
        if not order:
            chosen=int(feasible[np.lexsort((ids[feasible],-confidence[ids[feasible]]))[0]])
        else:
            best=None;chosen=None
            for start in range(0,len(feasible),512):
                ix=feasible[start:start+512];q=spectrum(np,A[ids[ix]]+matrix[None])[4];evaluations+=len(ix)
                rounded=np.fromiter((round(float(t),12) for t in q),dtype=np.float64,count=len(q))
                winner=int(np.lexsort((ids[ix],-confidence[ids[ix]],-rounded))[0]);j=int(ix[winner]);key=(-float(rounded[winner]),-float(confidence[ids[j]]),int(ids[j]))
                if best is None or key<best:best,chosen=key,j
        row=int(ids[chosen]);matrix+=A[row];done[chosen]=True;used[cellpos[chosen]]+=1;order.append(row)
        stat=information(np,matrix,len(order));history.append([stat['Q'],stat['absolute_gap'],stat['absolute_gap_per_row'],stat['coplanarity'],stat['trace']])
    need(np.array_equal(used,pol['quotas']),'all quotas exactly complete even negative gains')
    return dict(ids=np.asarray(sorted(order),dtype=np.int64),order=np.asarray(order,dtype=np.int64),history=np.asarray(history,dtype=np.float64).reshape(k,5),information=information(np,matrix,k),evaluations=evaluations)

def predictions(np,bank,capture,record,models,freeze):
    n=len(bank['star0']);score=capture['v_scores'];confidence=capture['v_confidence'];mask=eligible(np,score,confidence,CUTS);ids=np.flatnonzero(mask).astype(np.int64);k=min(target_count(n),len(ids))
    pol=policy(np,bank['star0'][ids],bank['star1'][ids],record['hw0'],record['hw1'],k)
    if all(bool(m['success']) for m in models):
        geo=geometry(np,bank,capture,record,models,freeze['normal_cap']);info=information(np,geo['contributions'][ids].sum(axis=0),len(ids));decision=reliable(geo['metrics'],info,freeze['bounds'])
    else:
        info=information(np,np.zeros((3,3),dtype=np.float64),0);decision=reliable(model_metrics(np,models),info,freeze['bounds']);geo=None
    rng=np.random.default_rng(20261031+1009*int(record['pair_index']));random_perm=rng.permutation(len(ids));priorities=np.empty(len(ids),dtype=np.float64);priorities[random_perm]=np.arange(len(ids),0,-1)
    perm=np.arange(n,dtype=np.int64)
    for cell in pol['occupied']:
        where=ids[pol['cells']==cell];perm[where]=where[rng.permutation(len(where))]
    fallback=not decision['accepted'];allids=np.arange(n,dtype=np.int64);C=None;G=None
    if fallback:subsets=[allids.copy() for _ in range(8)]
    else:
        A=geo['contributions'];C=greedy(np,A,ids,pol,confidence,k);G=greedy(np,A[perm],ids,pol,confidence,k)
        subsets=[allids,C['ids'],ids[select_cells(np,confidence[ids],pol)],ids[select_cells(np,score[ids],pol)],ids[select_cells(np,priorities,pol)],G['ids'],rank(np,confidence,k),rank(np,score,k)]
    subsets.extend([allids,np.arange(len(bank['stock0']),dtype=np.int64)])
    arrays=dict(K0=np.asarray(record['K0'],dtype=np.float64),K1=np.asarray(record['K1'],dtype=np.float64),hw0=np.asarray(record['hw0'],dtype=np.int64),hw1=np.asarray(record['hw1'],dtype=np.int64),v_all_row_ids=bank['acquired_row_ids'].copy(),v_scores=score.copy(),v_confidence=confidence.copy(),v_eligible=mask,v_quality_ids=ids,v_target_count=np.asarray(target_count(n),dtype=np.int64),v_retained_count=np.asarray(k,dtype=np.int64),v_fallback=np.asarray(fallback),v_geometry_permutation=perm,v_random_permutation=random_perm)
    arrays.update({'v_'+key:value for key,value in pol.items()})
    if geo is not None:arrays.update({'v_'+key:geo[key] for key in ('normals','normal_norms','capped_normals','opposite_parity_residual','weights','contributions')})
    for label,selected in [('C',C),('G',G)]:
        if selected is not None:arrays.update({label+'_order':selected['order'],label+'_history':selected['history']})
    diagnostics={}
    for m,ix in enumerate(subsets):
        stem='star' if m==8 else 'stock' if m==9 else 'primary'
        need(ix.dtype==np.int64 and bool(np.all(ix[1:]>ix[:-1])),'original sorted unique subset')
        for side in (0,1):arrays[f'm{m}_p{side}']=bank[f'{stem}{side}'][ix].copy()
        arrays[f'm{m}_indices']=ix
        if 1<=m<=7:need(len(ix)==(n if fallback else k),'shared fallback or exact count')
        if geo is not None and 1<=m<=7:diagnostics[ARMS[m]]=information(np,geo['contributions'][ix].sum(axis=0),len(ix))
    return arrays,dict(rows=n,eligible=len(ids),k=k,retained=len(subsets[1]),fallback=fallback,decision=decision,whole_eligible_information=info,selected_information_DIAGNOSTIC_ONLY=diagnostics,metrics=geo['metrics'] if geo else model_metrics(np,models),C_equals_B=np.array_equal(subsets[1],subsets[0]),C_equals_margin=np.array_equal(subsets[1],subsets[3]),C_evaluations=0 if C is None else C['evaluations'],G_evaluations=0 if G is None else G['evaluations'])
