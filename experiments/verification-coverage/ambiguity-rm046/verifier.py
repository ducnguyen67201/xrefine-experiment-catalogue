"""RM046: frozen-feature ambiguity, no learned parameters or coordinate edits."""
import math

def need(x,s):
    if not x:raise ValueError(s)

def retained_count(n):
    return min(n,max(5,(n+1)//2))

def selected(np,score,k):
    need(score.ndim==1 and np.isfinite(score).all() and 0<=k<=len(score),'finite rank/select')
    return np.sort(np.lexsort((np.arange(len(score)), -score))[:k]).astype(np.int64)

def margin(torch,a,b):
    """Minimum bidirectional own-partner minus best OTHER partner cosine."""
    need(a.ndim==b.ndim==2 and a.shape==b.shape,'same acquired populations')
    n=len(a)
    if n<2:return torch.zeros(n,dtype=a.dtype)
    score=a@b.T
    own=score.diag().clone();score.fill_diagonal_(-float('inf'))
    return torch.minimum(own-score.max(dim=1).values,own-score.max(dim=0).values)

def spatial(np,p0,p1,hw0,hw1,k):
    """GT-free farthest-point subset in normalized joint4D coordinate space."""
    n=len(p0)
    if k==0:return np.empty(0,dtype=np.int64)
    x=np.column_stack((p0/np.asarray(hw0[::-1]),p1/np.asarray(hw1[::-1]))).astype(np.float64)
    chosen=[int(np.argmin(np.sum((x-x.mean(axis=0))**2,axis=1)))]
    distance=np.sum((x-x[chosen[0]])**2,axis=1)
    distance[chosen]=-1
    while len(chosen)<k:
        i=int(np.argmax(distance));chosen.append(i)
        distance=np.minimum(distance,np.sum((x-x[i])**2,axis=1));distance[chosen]=-1
    return np.sort(np.asarray(chosen,dtype=np.int64))

def extract(torch,extractor,images,p0,p1,scales):
    """Sample frozen dense maps on ACTUAL acquired endpoints, never sparse bank.

    Base and relative-scale samples use exact vendor preprocessing + bicubic
    sparse interpolation. Returned images/points are never solver coordinates.
    """
    import torch.nn.functional as F
    def sample(image,points,scale):
        h,w=image.shape[-2:]
        resized=image[None] if scale==1 else F.interpolate(image[None],scale_factor=scale,mode='bilinear',align_corners=False)
        rh,rw=resized.shape[-2:]
        x,sy,sx=extractor.preprocess_tensor(resized)
        feat,_,_=extractor.net(x);feat=F.normalize(feat,dim=1)
        pos=torch.as_tensor(points.copy(),dtype=torch.float32)[None]
        pos=pos*torch.tensor([rw/w/sx,rh/h/sy])
        out=F.normalize(extractor.interpolator(feat,pos,H=x.shape[-2],W=x.shape[-1]),dim=-1)[0]
        return out
    with torch.inference_mode():
        f0=sample(images[0],p0,1.)
        fs=[sample(images[1],p1,s) for s in scales]
        scores=torch.stack([margin(torch,f0,f1) for f1 in fs])
        valid=torch.as_tensor((p0[:,0]>=0)&(p0[:,0]<images[0].shape[-1])&(p0[:,1]>=0)&(p0[:,1]<images[0].shape[-2])&(p1[:,0]>=0)&(p1[:,0]<images[1].shape[-1])&(p1[:,1]>=0)&(p1[:,1]<images[1].shape[-2]))
        scores[:,~valid]=-4.
    need(torch.isfinite(scores).all().item(),'finite scores')
    return scores.numpy().copy(),[f0.numpy().copy()]+[f.numpy().copy() for f in fs]

def native_confidence(torch,np,extractor,images,expected):
    """Observe original fine logits and replay original acquired coordinates.

    Existing code runs unchanged; hook captures the actual fine matcher output.
    Only byte-identical coordinates may bind confidence rows to saved matches.
    """
    captures=[]
    handle=extractor.net.fine_matcher.register_forward_hook(lambda module,args,out:captures.append(out.detach().clone()))
    try:
        with torch.inference_mode():fresh=extractor.match_xfeat_star(images[0][None],images[1][None],top_k=10000)
    finally:handle.remove()
    need(len(captures)==1 and all(np.array_equal(x,y) and x.dtype==y.dtype for x,y in zip(fresh,expected)),'byte-parity native reacquisition; confidence row binding')
    confidence=torch.softmax(captures[0]*3,dim=-1).max(dim=-1).values
    confidence=confidence[confidence>.25].numpy().copy()
    need(confidence.shape==(len(expected[0]),) and np.isfinite(confidence).all(),'fine-confidence selected row alignment')
    logits=captures[0][torch.softmax(captures[0]*3,dim=-1).max(dim=-1).values>.25].numpy().copy()
    return confidence,logits

def predictions(np,bank,scores,confidence,record):
    n=len(bank['star0']);k=retained_count(n)
    need(scores.shape==(3,n) and confidence.shape==(n,),'three frozen scales and confidence rows')
    score=scores.min(axis=0);seed=20261030+int(record['pair_index'])*1009
    rng=np.random.default_rng(seed)
    random_ids=np.sort(rng.permutation(n)[:k]).astype(np.int64)
    shuffled=score[rng.permutation(n)]
    ids=[np.arange(n,dtype=np.int64),selected(np,score,k),selected(np,confidence,k),random_ids,
         spatial(np,bank['star0'],bank['star1'],record['hw0'],record['hw1'],k),selected(np,shuffled,k),
         np.arange(n,dtype=np.int64),np.arange(len(bank['stock0']),dtype=np.int64)]
    arrays=dict(K0=np.asarray(record['K0'],dtype=np.float64),K1=np.asarray(record['K1'],dtype=np.float64),hw0=np.asarray(record['hw0'],dtype=np.int64),hw1=np.asarray(record['hw1'],dtype=np.int64),v_scores=scores,v_confidence=confidence,v_shuffled_score=shuffled,v_all_row_ids=bank['acquired_row_ids'].copy())
    for m,ix in enumerate(ids):
        stem='star' if m==6 else 'stock' if m==7 else 'primary'
        for side in (0,1):arrays[f'm{m}_p{side}']=bank[f'{stem}{side}'][ix].copy()
        arrays[f'm{m}_indices']=ix
    need(all(len(ids[m])==k for m in range(1,6)),'equal-count verifier and filtered controls')
    return arrays
