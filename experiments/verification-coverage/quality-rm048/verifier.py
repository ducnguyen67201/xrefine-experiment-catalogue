"""RM048: quality-gated coverage of existing acquired rows; no weak fill."""
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

def selections(np,p0,p1,hw0,hw1,score,conf,cuts,pair_index):
    mask=eligible(np,score,conf,cuts);ids=np.flatnonzero(mask).astype(np.int64)
    n=len(score);k=min(target_count(n),len(ids))
    p=policy(np,p0[ids],p1[ids],hw0,hw1,k)
    rng=np.random.default_rng(20261030+1009*int(pair_index))
    within=rng.permutation(len(ids));joint=rng.permutation(n);random=rng.permutation(n)
    ws=score.copy();ws[ids]=score[ids][within]
    js,jc=score[joint],conf[joint]
    jm=eligible(np,js,jc,cuts);ji=np.flatnonzero(jm).astype(np.int64)
    need(len(ji)==len(ids),'joint shuffle preserves eligibility count')
    jp=policy(np,p0[ji],p1[ji],hw0,hw1,k)
    out=[np.arange(n,dtype=np.int64),ids[select_cells(np,score[ids],p)],rank(np,conf,k),
         np.sort(random[:k]).astype(np.int64),ids[select_cells(np,ws[ids],p)],
         ji[select_cells(np,js[ji],jp)],rank(np,score,k),ids[rank(np,score[ids],k)]]
    need(all(len(x)==k for x in out[1:]),'ALL filtered equal counts including zero')
    need(all(np.all(mask[x]) for x in (out[1],out[4],out[7])),'no ineligible weak-cell fill')
    if len(ids)<=target_count(n):
        need(np.array_equal(out[1],out[7]),'identical when all eligible fit; no method effect')
    meta=dict(v_eligible=mask,v_quality_ids=ids,v_joint_eligible=jm,v_joint_quality_ids=ji,
        v_within_permutation=within,v_joint_permutation=joint,v_random_permutation=random,
        v_within_shuffled_score=ws,v_joint_shuffled_score=js,v_joint_shuffled_confidence=jc,
        v_margin_cut=np.asarray(cuts['margin'],dtype=np.float64),
        v_conf_cut=np.asarray(cuts['confidence'],dtype=np.float64),
        v_target_count=np.asarray(target_count(n),dtype=np.int64),
        v_retained_count=np.asarray(k,dtype=np.int64))
    meta.update({'v_'+a:b for a,b in p.items()})
    meta.update({'v_joint_'+a:b for a,b in jp.items()})
    return out,meta

def predictions(np,bank,capture,record,cuts):
    n=len(bank['star0'])
    ids,meta=selections(np,bank['star0'],bank['star1'],record['hw0'],record['hw1'],
                       capture['v_scores'],capture['v_confidence'],cuts,record['pair_index'])
    ids.extend([np.arange(n,dtype=np.int64),np.arange(len(bank['stock0']),dtype=np.int64)])
    arrays=dict(capture,**meta,K0=np.asarray(record['K0'],dtype=np.float64),
        K1=np.asarray(record['K1'],dtype=np.float64),hw0=np.asarray(record['hw0'],dtype=np.int64),
        hw1=np.asarray(record['hw1'],dtype=np.int64),v_all_row_ids=bank['acquired_row_ids'].copy())
    for m,ix in enumerate(ids):
        stem='star' if m==8 else 'stock' if m==9 else 'primary'
        for side in (0,1):arrays[f'm{m}_p{side}']=bank[f'{stem}{side}'][ix].copy()
        arrays[f'm{m}_indices']=ix
    return arrays
