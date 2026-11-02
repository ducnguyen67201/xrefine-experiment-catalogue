"""RM022 finite RGB context binding; caller must be the admitted COPY owner.

No old B15 manager, main, actor, tracking gate or lifecycle is imported here.
The nested science definitions are unchanged extracted ASTs; file/slice evidence
is outside COPY. Only the owner may call this factory after genuine runtime,
source, process and journal admission. SOURCE publication is not execution.
"""
import ast
import base64
import collections.abc as collections
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Optional, Tuple


def bind_context(req, *, canonical_root, plan, criteria, runtime,
                 frames, source_packets, source_pin, current_sources,
                 report_read, origin_storage, append):
    """Bind exact science to the owner's real COPY callbacks, never mocks.

    The seven callback arguments are unresolved host operands until the owner
    supplies their existing qualified COPY source/path/symbol and pins. This
    function creates no process, reservation, journal, writer or run loop.
    `append` must authenticate and durably CAS req['report_prefix']; `frames`
    must validate/project the same attempt with source/nonce/order identities.
    Runtime modules must already have been admitted by COPY's real observer.
    """
    ROOT = Path(canonical_root)
    MIB = 1048576
    PLAN = plan
    PRE = criteria
    def need(v,label):
     if not v:raise RuntimeError(label)

    def enc(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')

    def sha(b):return hashlib.sha256(b).hexdigest()

    def packed(b):return dict(bytes=len(b),sha256=sha(b),zlib_base64=base64.b64encode(zlib.compress(b,6)).decode('ascii'))

    def unpack(q,cap=8*MIB):
     need(type(q)==dict and set(q)=={'bytes','sha256','zlib_base64'} and type(q['bytes']) is int and 0<=q['bytes']<=cap and type(q['sha256']) is str and re.fullmatch('[0-9a-f]{64}',q['sha256']) and type(q['zlib_base64']) is str,'strict packed types')
     need(len(q['zlib_base64'])<=4*((cap+cap//1000+130)//3),'encoded cap')
     d=zlib.decompressobj();b=d.decompress(base64.b64decode(q['zlib_base64'],validate=True),cap+1)
     need(d.eof and not d.unused_data and not d.unconsumed_tail and len(b)==q['bytes'] and sha(b)==q['sha256'],'bounded exact inflate');return b

    def clean_json(b):
     def pairs(items):
      d={}
      for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
      return d
     return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

    def latest(schema):
     rows=[r for r in frames() if r['schema']==schema];need(rows,'missing '+schema);return rows[-1]

    def strict_arrays(payload,selected=None):
     raw=payload if type(payload) is bytes else unpack(payload,8*MIB);need(len(raw)<=8*MIB and (selected is None or type(selected) is set),'NPZ cap/selection');out={}
     with zipfile.ZipFile(io.BytesIO(raw)) as z:
      infos=z.infolist();need(0<len(infos)<=32 and len({x.filename for x in infos})==len(infos) and sum(x.file_size for x in infos)<=8*MIB,'finite unique NPZ')
      for x in infos:
       need(re.fullmatch('[A-Za-z][A-Za-z0-9_]*[.]npy',x.filename) and x.file_size<=8*MIB and not x.flag_bits&1,'plain array member')
       if selected is not None and x.filename[:-4] not in selected:continue
       b=z.read(x);need(b[:6]==b'\x93NUMPY' and b[6:8] in (b'\x01\x00',b'\x02\x00',b'\x03\x00'),'NPY version')
       sz=2 if b[6]==1 else 4;n=int.from_bytes(b[8:8+sz],'little');need(n<=4096,'header cap');h=ast.literal_eval(b[8+sz:8+sz+n].decode('latin1'))
       need(type(h) is dict and set(h)=={'descr','fortran_order','shape'} and h['fortran_order'] is False and type(h['shape']) is tuple and len(h['shape'])<=4 and all(type(q) is int and 0<=q<=1000000 for q in h['shape']),'typed shape');dt=h['descr'];need(dt in {'<f8','<f4','<i8','<i4','|u1','|b1','<u8'},'plain dtype');fmt={'<f8':'d','<f4':'f','<i8':'q','<i4':'i','|u1':'B','|b1':'?','<u8':'Q'}[dt];data=b[8+sz+n:]
       need(len(data)==math.prod(h['shape'])*struct.calcsize('<'+fmt),'exact payload');out[x.filename[:-4]]=dict(shape=h['shape'],dtype=dt,values=[q[0] for q in struct.iter_unpack('<'+fmt,data)],raw=data)
     need(selected is None or set(out)==selected,'all selected fields present');return out

    class API:
     def __init__(self,req):
      self.req=req;self.plan=PLAN;self.need=need;self.enc=enc;self.sha=sha;self.readpin=readpin;self.packed=packed;self.unpack=unpack;self.frames=frames;self.latest=latest;self.gt_pin=PRE['auditor_only_gt'];self.np=None;self.torch=None;self.cv2=None
     def __getitem__(self,key):return getattr(self,key)
     def emit(self,value=None,**kwargs):
      if isinstance(value,str):v=dict(schema=value,**kwargs)
      else:need(type(value) is dict and not kwargs,'emit object');v=dict(value)
      role=self.req['role'];allowed={'producer':{'guidance-pair-v1','guidance-seal-v1'},'audit':{'audit-v1'},'reference':{'root-confirmation-v1','reference-mismatch-v1'}}
      diagnostic={'native-before-v1','native-after-v1','native-failure-v1','python-origins-before-v1','python-origins-after-v1','python-origins-failure-v1'};need(v.get('schema') in allowed.get(role,set())| (diagnostic if role in {'producer','audit'} else set()),'role/schema allowlist');q=source_packets();identity=dict(nonce=self.req['nonce'],role=role,source=dict(bytes=q[role]['bytes'],sha256=q[role]['sha256']),controller_source=source_pin(),sources=current_sources())
      if v['schema'] in {'root-confirmation-v1','reference-mismatch-v1'}:identity['audit_payload_sha256']=latest('audit-v1')['payload']['sha256']
      for k,x in identity.items():need(k not in v or v[k]==x,'no conflicting artifact provenance');v[k]=x
      if v['schema']=='guidance-pair-v1':need(v['payload']['bytes']<=MIB,'pair raw bound')
      if v['schema']=='audit-v1':need(v['payload']['bytes']<=8*MIB and len(enc(v.get('meta',{})))<=MIB,'score bounds')
      if v['schema']=='root-confirmation-v1':unpack(v['payload'],8*MIB)
      origin_storage(report_read(),v)
      rp=append(v,self.req['report_prefix']);self.req['report_prefix']=rp;return rp
     def pack_arrays(self,m):
      need(self.np is not None and type(m) is dict and 0<len(m)<=32,'numeric encoder')
      for k,a in m.items():need(type(k) is str and re.fullmatch('[A-Za-z][A-Za-z0-9_]*',k) and isinstance(a,self.np.ndarray) and not a.dtype.hasobject and a.dtype.str in {'<f8','<f4','<i8','<i4','|u1','|b1','<u8'},'plain array')
      f=io.BytesIO();self.np.savez_compressed(f,**m);raw=f.getvalue();strict_arrays(raw);return raw
     def decode_arrays(self,payload,numeric=True):
      need(type(numeric) is bool,'numeric flag');d=strict_arrays(payload)
      if not numeric:return {k:{x:v[x] for x in ('shape','dtype','values')} for k,v in d.items()}
      need(self.np is not None,'numeric runtime');return {k:self.np.frombuffer(v['raw'],dtype=v['dtype']).copy().reshape(v['shape']) for k,v in d.items()}
     def arrayhash(self,a):return sha(enc([a.dtype.str,list(a.shape)])+a.tobytes(order='C'))
     def points(self,pair):
      need(self.np is not None and pair in PLAN['pairs'],'fixed pair');q=pair['points']['packed']
      raw=readpin(q['packed']['container'])
      with zipfile.ZipFile(io.BytesIO(raw)) as z:
       names=z.namelist();need(names.count(q['packed']['member'])==1,'unique bank member');b=z.read(q['packed']['member'])
      need(len(b)==q['bytes'] and sha(b)==q['sha256'],'whole bank/member SHA/CRC');d=strict_arrays(b,{'primary0','primary1','acquired_row_ids'});need(set(d)=={'primary0','primary1','acquired_row_ids'},'three selected fields')
      arrays={k:self.np.frombuffer(v['raw'],dtype=v['dtype']).copy().reshape(v['shape']) for k,v in d.items()};a,c,ids=[arrays[k] for k in ['primary0','primary1','acquired_row_ids']]
      need(a.shape==c.shape==(pair['star_matches'],2) and ids.shape==(pair['star_matches'],) and a.dtype==c.dtype==self.np.float32 and ids.dtype.str=='<i8' and self.np.isfinite(a).all() and self.np.isfinite(c).all(),'fixed points typed population');return a,c,ids
     def image(self,pair,side):
      need(type(side) is int and side in (0,1) and pair in PLAN['pairs'],'known image');q=pair['image_pins'][side];readpin(q)
      x=self.pixel['load_image'](path(q['path']));data=self.processor(x);v=self.pixel['unpad_image'](data);K=data['transform']@self.np.asarray(pair['original_K'+str(side)],dtype=self.np.float64)
      self.pixel['validate_processed'](v,K,pair['hw'+str(side)],pair['K'+str(side)]);qx=[x for x in PLAN['images'] if x['metadata']['path']==q['path']];need(len(qx)==1 and self.pixel['pixel_signature'](v)==qx[0]['pixels_sha256'],'exact saved pixels');readpin(q);return v
     def truth(self):
      need(self.req['role']=='audit','GT parse audit only');validate_guidance(self.req['nonce']);return clean_json(readpin(self.gt_pin))

    def render(raw,names,env):
     tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
     need({n.name for n in nodes}==set(names),'exact selected definition closure');exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-rgb-leaves>','exec'),env)

    def provenance(row,role):
     q=source_packets()[role];need(row['nonce']==latest('attempt-v1')['nonce'] and row['role']==role and row['source']==dict(bytes=q['bytes'],sha256=q['sha256']) and row['controller_source']==source_pin() and row['sources']==current_sources(),'exact saved artifact source joins')

    def validate_guidance(nonce):
     rows=frames();parts=[r for r in rows if r['schema']=='guidance-pair-v1'];seal=latest('guidance-seal-v1');provenance(seal,'producer')
     need(len(parts)==32 and [r['pair_index'] for r in parts]==[q['pair_index'] for q in PLAN['pairs']] and sum(r['payload']['bytes'] for r in parts)<=8*MIB,'complete fixed ordered population');ordered=[]
     for r,q in zip(parts,PLAN['pairs']):
      provenance(r,'producer');need(r['nonce']==seal['nonce']==nonce and r['count']==q['star_matches'] and r['safeplan_sha']==sha(enc(PLAN)),'guidance source population')
      d=strict_arrays(r['payload']);need(set(d)=={'G','C','flags'} and d['G']['shape']==d['C']['shape']==(q['star_matches'],2,3) and d['G']['dtype']==d['C']['dtype']=='<f8' and d['flags']['shape']==(q['star_matches'],2) and d['flags']['dtype']=='|u1','guidance shape')
      need(all(math.isfinite(x) for k in ['G','C'] for x in d[k]['values']) and all(x<=3 for x in d['flags']['values']),'finite complete guidance');expected=[next(x['pixels_sha256'] for x in PLAN['images'] if x['metadata']['path']==name) for name in q['image_paths']];need(r['pixel_hashes']==expected,'pixel signatures')
      p=q['points']['packed']
      with zipfile.ZipFile(io.BytesIO(readpin(p['packed']['container']))) as z:raw=z.read(p['packed']['member'])
      need(len(raw)==p['bytes'] and sha(raw)==p['sha256'],'original correspondence member pin');a=strict_arrays(raw,{'primary0','primary1','acquired_row_ids'})
      need(r['input_hashes']=={k:sha(enc([v['dtype'],list(v['shape'])])+v['raw']) for k,v in a.items()},'all unchanged input digests');ordered.append([q['pair_index'],r['payload']['bytes'],r['payload']['sha256']])
     need(seal['ordered_payloads']==ordered and seal['pairs']==32 and seal['cases']==68578 and seal['safeplan_sha']==sha(enc(PLAN)) and type(seal['fixtures']) is list and len(seal['fixtures'])>=10 and all(seal[k] is True for k in ['zero_GT_pose_reads','zero_NN_calls','zero_native_GC_calls','all_rows_retained']),'complete guidance seal');return seal

    def saved_audit(nonce):
     r=latest('audit-v1');provenance(r,'audit');need(r['nonce']==nonce,'audit nonce');d=strict_arrays(r['payload']);N=68578
     need(set(d)=={'delta','q','nll','pit','perm','valid'} and all(d[k]['dtype']=='<f8' for k in ['delta','q','nll','pit']) and d['delta']['shape']==(N,) and all(d[k]['shape']==(N,4) for k in ['q','nll','pit']) and d['perm']['shape']==d['valid']['shape']==(N,) and d['perm']['dtype']=='<i8' and d['valid']['dtype']=='|u1','all saved score arrays');meta=r['meta'];need(meta['cases']==N and meta['pairs']==32 and meta['arms']==['gradient','constant','trace','shuffle'] and meta['no_new_pose_accuracy'] is True and meta['forwards']==meta['new_GC_fits']==0 and meta['outcome'] in {'QUALIFIED_PROXY','SCIENTIFIC_NEGATIVE','NO_QUALIFICATION'},'score metadata scope')
     start=0
     for q,s in zip(PLAN['pairs'],meta['pair_slices']):
      need(s['pair_index']==q['pair_index'] and s['scene']==q['scene'] and s['count']==q['star_matches'] and s['start']==start and s['hw0']==q['hw0'] and s['hw1']==q['hw1'],'ordered score slices');n=q['star_matches'];p=d['perm']['values'][start:start+n]
      need(sorted(p)==list(range(n)),'all permutation indices');start+=n
     need(len(meta['pair_slices'])==32 and start==N and all(x in (0,1) for x in d['valid']['values']) and meta['undefined_count']==d['valid']['values'].count(0),'all scored or undefined rows')
     for i,valid in enumerate(d['valid']['values']):
      if valid:need(math.isfinite(d['delta']['values'][i]) and all(math.isfinite(x) and x>0 for x in d['q']['values'][4*i:4*i+4]),'defined score geometry')
      else:need(math.isnan(d['delta']['values'][i]) and all(math.isnan(x) for k in ['q','nll','pit'] for x in d[k]['values'][4*i:4*i+4]),'undefined rows retained NaN')
     need(type(meta['fixtures']) is list and len(meta['fixtures'])>=10,'auditor authored fixture results');return r

    def saved_reference(nonce,reply):
     ar=saved_audit(nonce);r=latest('root-confirmation-v1');provenance(r,'reference');value=clean_json(unpack(r['payload'],2*MIB));need(enc(value)==unpack(r['payload'],2*MIB),'canonical reference result')
     need(r['nonce']==nonce and r['audit_payload_sha256']==ar['payload']['sha256'] and r['complete'] is True and value['schema']=='rgb-gradient-scalar-reference-v1' and value['complete'] is True and value['status']=='PASS' and value['cases']==68578 and value['pairs']==32,'complete independent saved confirmation');need(value['audit_outcome']==ar['meta']['outcome'] and value['outcome']==r['outcome']==reply['outcome'] and value['advance']==r['advance']==reply['advance']==(value['outcome']=='QUALIFIED_PROXY'),'saved decision/advance joins');need(value['undefined_count']==ar['meta']['undefined_count'] and value['no_new_pose_accuracy'] is True and value['no_model_calls'] is True and value['maximum_differences']==reply['maximum_differences'] and value['boundary_disagreements']==reply['boundary_disagreements'],'all ROOT summary joins');need(value['fixtures']['complete'] is True and value['fixtures']['accepted']==len(value['fixtures']['labels']) and value['fixtures']['accepted']>=10 and len(unpack(value['boundary_mask'],4*68578))==4*68578,'complete reference fixtures/boundary mask')
     if value['advance']:need(len(value['gates'])==len(ar['meta']['gates'])==14 and all(g['status']=='PASS' for g in value['gates']+ar['meta']['gates']) and value['boundary_disagreements']==0 and not value['gate_disagreements'],'both fixed gate sets pass')
     return value

    def path(name):
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        need(str(p).startswith('/') and not str(p).startswith('//') and '..' not in p.parts,'canonical path')
        need(not any(q.is_symlink() for q in (p,*p.parents)),'no symlink ancestry')
        return p

    def readpin(q,cap=256*MIB,links=1):
        need(isinstance(q,dict) and type(q['bytes']) is int and q['bytes']>=0 and isinstance(q['path'],str) and re.fullmatch('[0-9a-f]{64}',q['sha256']) is not None,'strict pin types')
        p=path(q['path']);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            a=os.fstat(fd);need(stat.S_ISREG(a.st_mode) and a.st_nlink==links and a.st_size==q['bytes']<=cap,'pinned shape/link/size')
            chunks=[];h=hashlib.sha256();n=0
            while True:
                b=os.read(fd,MIB)
                if not b:break
                n+=len(b);need(n<=cap,'read cap');h.update(b);chunks.append(b)
            z=os.fstat(fd);t=p.stat();sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
            need(sig(a)==sig(z)==sig(t) and n==q['bytes'] and h.hexdigest()==q['sha256'],'pinned stable hash')
            return b''.join(chunks)
        finally:os.close(fd)

    need(type(req) is dict and req.get('role') in {'producer','audit','reference'},
         'existing scientific role only; no old probe or actor')
    need(type(req.get('nonce')) is str and len(req['nonce']) == 32 and
         req['nonce'] == latest('attempt-v1')['nonce'], 'same admitted COPY attempt')
    need(type(req.get('data_deadline')) in (int,float) and
         math.isfinite(req['data_deadline']), 'original absolute monotonic deadline')
    need(type(req.get('report_prefix')) is dict, 'actual durable append prefix')
    need(len(enc(PLAN)) == 123837 and
         sha(enc(PLAN)) == '0b116eb58546e137e663b546ecb44828ce03f8030ed889ce9247a23f04bf02af',
         'unchanged complete safe plan')
    need(PLAN['counts'] == {'pairs':32,'images':57,'matches':68578,'max_matches':5580},
         'original fixed qualification population')
    need(len(enc(PRE)) == 5586 and
         sha(enc(PRE)) == 'a1b47e378d717451e82276cbfc259ae4e3847a55edacad7f99e408fb728b52c2',
         'exact unchanged criteria/source-pin metadata')
    need(PRE['namespace'] == 'rgb_gradient_uncertainty_v1' and
         PRE['arms'] == ['gradient','constant','trace','shuffle'], 'original science scope')
    need(PRE['auditor_only_gt'] == {
        'path':'vendor/XFeat/assets/megadepth_1500.json','bytes':4024939,
        'sha256':'6aa01108aa26b1d5f5f1aad58a35514a173febd525f71cdab312ef72edbcca19'},
        'unchanged lazy audit-only truth descriptor')
    packets = source_packets()
    expected = {
        'producer':(17165,'5e958601f86215dc3e96a9541b79e5b775d52ac94923f657b9d6c4da6d81e8c3'),
        'audit':(18888,'2e16471cfd0912dbe458fa36dd96198a3d4937fc4b6faad902a3a0ac5f99f4cd'),
        'reference':(22718,'b96dabc411d8a6c9563d43e2c4d34bbadf4d58b49170b2ebbf1b97ae3c6a4059')}
    need(set(packets) == {'controller',*expected}, 'actual complete source role roster')
    for role,(n,h) in expected.items():
        need((packets[role]['bytes'],packets[role]['sha256']) == (n,h),
             'unchanged loaded P/A/R body pins')
    controller = packets['controller']
    need(type(controller['bytes']) is int and 0 < controller['bytes'] <= 128*1024 and
         controller['sha256'] != 'cbc8221d613c18a205ddbf39dedc202e25912ed25992f0decb79a89741881441',
         'actual finite COPY controller; never identify as whole old B15')
    need(current_sources() == [dict(role=r,bytes=packets[r]['bytes'],sha256=packets[r]['sha256'])
                              for r in ('controller','producer','audit','reference')],
         'ordered actual loaded source roster')
    cp = source_pin()
    need(cp['bytes'] == controller['bytes'] and cp['sha256'] == controller['sha256'] and
         type(cp.get('enrollment_bytes')) is int and cp['enrollment_bytes'] > 0 and
         type(cp.get('enrollment_sha256')) is str and
         re.fullmatch('[0-9a-f]{64}',cp['enrollment_sha256']),
         'actual COPY source and frozen source-prefix identity')
    context = API(req)
    context.validate_guidance = validate_guidance
    context.saved_audit = saved_audit
    context.saved_reference = saved_reference
    if req['role'] in {'producer','audit'}:
        # Only scientific pixel setup from old lines954-958 follows. The
        # COPY owner supplies already observed modules; no numeric imports,
        # sys.path mutation, thread/seed mutation or old runtime lifecycle.
        need(type(runtime) is dict and set(runtime) == {'np','torch','cv2','kornia','OmegaConf'},
             'exact already-qualified COPY numeric runtime operands')
        np,torch,cv2,kornia,OmegaConf = (runtime[k] for k in
                                     ('np','torch','cv2','kornia','OmegaConf'))
        need(not torch.cuda.is_available() and torch.get_num_threads() == 4 and
             torch.get_num_interop_threads() == 1 and cv2.getNumThreads() == 1,
             'owner-established CPU/four compute/interop1/CVserial runtime')
        A = context
        A.np=np;A.torch=torch;A.cv2=cv2
        env=dict(np=np,torch=torch,cv2=cv2,kornia=kornia,collections=collections,Optional=Optional,Tuple=Tuple,OmegaConf=OmegaConf,Path=Path,hashlib=hashlib,json=json,need=need);pins={Path(q['path']).name:q for q in PRE['sources']};need(len(pins)==len(PRE['sources']),'unique renderer source names')
        render(readpin(pins['image.py']),{'ImagePreprocessor','read_image','numpy_image_to_torch','load_image'},env);render(readpin(pins['benchmark.py']),{'unpad_image'},env);guard=readpin(pins['image_guard.py']);assign=[n for n in ast.parse(guard).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='PREDICATES']
        need(len(assign)==1,'one guard literal');p=ast.literal_eval(assign[0].value);need(p==('cpu','fp32','chw3','hw','finite','nonnegative','processed_K','legacy_upper_ok'),'exact qualified predicates');env['PREDICATES']=p
        render(guard,{'inspect_processed','validate_processed','pixel_signature'},env);A.pixel=env;A.processor=env['ImagePreprocessor']({'resize':1024,'side':'long','square_pad':True})
    else:
        need(runtime is None, 'scalar reference has no numeric runtime binding')
    return context
