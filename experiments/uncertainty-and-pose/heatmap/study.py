"""Frozen decoder-shape qualification; unchanged author coordinates/weights.

Source leaves attributed to the preserved AR controller and author XRefine.
This file never dispatches an old experiment or imports its scientific actor.
"""
import ast,base64,contextlib,datetime,fcntl,hashlib,io,json,math,os,random,re,shutil,signal,stat,struct,subprocess,sys,time,traceback,types,uuid,zipfile,zlib
from pathlib import Path

ROOT=Path('/Users/ducng/Desktop/workspace/SLAMResaerch')
NS=ROOT/'research/localization-uncertainty-repair-20261024'
REPORT=NS/'report.md'
CODE=NS/'study.py'
KEY='localization_uncertainty_v2'
MIB=1048576
GIB=1024*MIB
SHARED=['research/managed-research-state.json','research/experiment-owner.lock/owner.json','.claude/plans/managed-mac-image-matching-goal.md']
TEMP=ROOT/'.claude/plans/localization-uncertainty-v2-shared.tmp'
OLD_FREEZE={'path':'research/prof-capture-basin-20261019/freeze-01.json','bytes':8682114,'sha256':'f01fd327e9a0d148fe18cd708a59fdf9e048278158dc14bd15b25c336d365396'}
INDEX={'path':'research/prof-crossview-joint-20261019/input-index.json','bytes':110654,'sha256':'4a9df838d9662c4c5949083c8009cf63d1e188af70c4e50d1eecbbac5f194095'}
HELPER={'path':'research/ar-persistence-20261014/controller-v8.py','bytes':37614,'sha256':'3088d315e314645b573454afb96c63f62e397574fa7a2e78b53b0d7b1a8c631b'}
REFSPEC={'path':'research/reference-rotation-20261024/spec.json','bytes':13848,'sha256':'e615adf58626261ef6cb1102322feba27d6b6dcbb66ce8f9236f01396cdd15bf'}
WEIGHT={'path':'vendor/XRefine/pretrained/xrefine_small_specific_xfeat-star.pth','bytes':1270242,'sha256':'5c91d3fad9d9b1b65b44ab4715d574ae53969641b862d99c418dceb765fbed04'}
PYTHON=ROOT/'benchmarks/xrefine/.venv/bin/python'
PYREAL='/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11'
BOOT=('import sys,os,hashlib,stat; p,n,h=sys.argv[1:4]; f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW); '
      's=os.fstat(f); b=os.read(f,int(n)+1); os.close(f); '
      'assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and len(b)==s.st_size==int(n) and hashlib.sha256(b).hexdigest()==h; '
      'sys.argv=[p,*sys.argv[4:]]; exec(compile(b,p,"exec"),{"__name__":"__main__","__file__":p})')
ENV=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',LANG='C',LC_ALL='C',TZ='UTC',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4',NUMEXPR_NUM_THREADS='4',NUMEXPR_MAX_THREADS='4',ECC_HOOK_PROFILE='strict')
CLOSED_V1={'path':'research/localization-uncertainty-20261024/report.md','bytes':2677705,'sha256':'b166aa9396bf9b1ab6a1508d447fb3cfb082ceb77aa3a30a532892a6a19f557a'}
RETIRED=('output/pdf/xrefine-progress-20261021/xrefine-progress-20261021.aux','output/pdf/xrefine-progress-20261021/xrefine-progress-20261021.out')
ARMS=('shape','constant','trace','shuffle')
LEVELS=(.5,.8,.9,.95)

def need(v,label):
    if not v:raise RuntimeError(label)

def enc(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

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

def actual(p,cap=16*MIB):
    p=path(p);cap=17*MIB if p==REPORT else cap;b=p.read_bytes();q=dict(path=str(p),bytes=len(b),sha256=sha(b));readpin(q,cap);return q

def frames(raw=None,closed=False):
    raw=REPORT.read_bytes() if raw is None else raw
    out=[];end=None;seen=set();pending=None;transactions=set();prepared=None;attempt_nonce=None;roles=set()
    for m in re.finditer(rb'<!-- RECORD -->\n```json\n([^\n]+)\n```\n',raw):
        if end is not None:need(raw[end:m.start()]==b'\n','strict complete frame tail')
        value=json.loads(m.group(1));need(enc(value).rstrip(b'\n')==m.group(1),'canonical frame')
        if 'previous_bytes' in value:
            n=value['previous_bytes'];need(type(n) is int and 0<=n<=m.start() and sha(raw[:n])==value['previous_sha256'],'frame chain')
            need(raw[n:m.start()] in (b'',b'\n'),'frame contiguous')
        schema=value['schema']
        if schema=='attempt-v1':attempt_nonce=value['nonce']
        if attempt_nonce is not None and 'nonce' in value:need(value['nonce']==attempt_nonce,'one exact attempt nonce')
        if schema in {'launch-attempt-v1','durable-child-v1','actor-observation-v1'}:
            rk=(schema,value['nonce'],value['role']);need(rk not in roles,'unique role frame');roles.add(rk)
        if schema in {'freeze-v1','attempt-v1','shared-originals-v1','model-raw-v1','audit-v1','root-confirmation-v1','terminal-receipt-v1','native-before-v2','model-plain-v2','native-after-v2','native-rejected-v2'}:
            need(schema not in seen,'duplicate scientific frame');seen.add(schema)
        if schema=='transaction-prepare-v1':
            key=(value['nonce'],value['tag']);need(pending is None and key not in transactions,'unique ordered PREPARE');pending=key;prepared=value
        if schema=='transaction-commit-v1':
            key=(value['nonce'],value['tag']);need(pending==key,'matching unique COMMIT')
            need([(q['bytes'],q['sha256']) for q in value['after']]==[(q['bytes'],q['sha256']) for q in prepared['after']] and all(Path(q['path'])==ROOT/n for q,n in zip(value['after'],SHARED)),'COMMIT exact PREPARE targets')
            transactions.add(key);pending=None;prepared=None
        out.append(value);end=m.end()
    need(out and out[0]['schema']=='uncertainty-prospective-v1','initial prospective record')
    need(raw[end:] in (b'',b'\n'),'no partial or foreign generated tail')
    need(not closed or pending is None,'unresolved PREPARE retains reservation')
    return out

def latest(schema):
    rows=[r for r in frames() if r['schema']==schema];need(rows,'missing '+schema);return rows[-1]

def append(value,expected=None):
    old=REPORT.read_bytes();need(len(old)<=17*MIB,'report cap before')
    if expected is not None:need(actual(REPORT)==expected,'exclusive frame prefix')
    frames(old);value=dict(value,previous_bytes=len(old),previous_sha256=sha(old))
    raw=b'\n<!-- RECORD -->\n```json\n'+enc(value).rstrip(b'\n')+b'\n```\n';need(len(old)+len(raw)<=17*MIB,'report cap after')
    a=REPORT.stat();need(a.st_nlink==1,'report ownership')
    fd=os.open(REPORT,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
    try:
        z=os.fstat(fd);need((a.st_dev,a.st_ino,a.st_size)==(z.st_dev,z.st_ino,z.st_size),'append identity')
        with os.fdopen(fd,'ab',closefd=False) as f:f.write(raw);f.flush();os.fsync(fd)
    finally:os.close(fd)
    need(REPORT.read_bytes()==old+raw,'append readback');frames();return actual(REPORT)

def packed(b):return dict(bytes=len(b),sha256=sha(b),zlib_base64=base64.b64encode(zlib.compress(b,6)).decode())

def unpack(v,cap):
    need(type(v['bytes']) is int and 0<=v['bytes']<=cap,'inflate declared cap')
    d=zlib.decompressobj();b=d.decompress(base64.b64decode(v['zlib_base64'],validate=True),cap+1)
    need(d.eof and not d.unused_data and not d.unconsumed_tail and len(b)==v['bytes'] and sha(b)==v['sha256'],'exact bounded inflate')
    return b

def old_leaves():
    # Compile only these unchanged definitions and their stdlib imports.
    raw=readpin(HELPER);tree=ast.parse(raw);names={'encode','utc','require','pin','physical','read','ps_snapshot','FrozenGate','supervise'}
    parts=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    need({n.name for n in parts if isinstance(n,(ast.FunctionDef,ast.ClassDef))}==names,'finite helper closure')
    class Strengthen(ast.NodeTransformer):
        def visit_Constant(self,n):
            return ast.copy_location(ast.Constant(4096),n) if n.value==131072 else ast.copy_location(ast.Constant(.2),n) if n.value==.08 else n
        def visit_Expr(self,n):
            n=self.generic_visit(n)
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='samples' and n.value.func.attr=='append':
                return [n,ast.parse("require(len(samples)<=launch_record['sample_limit'] and len(encode(samples[-1]))<=512,'finite observation samples')").body[0]]
            return n
        def visit_Assign(self,n):
            n=self.generic_visit(n)
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='subprocess' and n.value.func.attr=='Popen':
                return [ast.parse("launch_record['argv']=argv+[str(wfd),str(deadline)]").body[0],n]
            return n
    for n in parts:
        if isinstance(n,ast.FunctionDef) and n.name=='supervise':Strengthen().visit(n)
    d=dict(ROOT=ROOT,MIB=MIB,GIB=GIB,SITE=ROOT/'benchmarks/xrefine/.venv/lib/python3.11/site-packages',STDLIB=Path(PYREAL).parent.parent/'lib/python3.11',CLEAN_ENV=ENV)
    exec(compile(ast.fix_missing_locations(ast.Module(body=parts,type_ignores=[])),str(ROOT/HELPER['path']),'exec'),d)
    return types.SimpleNamespace(**d)

def roster():
    f=json.loads(readpin(OLD_FREEZE));idx=json.loads(readpin(INDEX));rs=json.loads(readpin(REFSPEC))
    need(len(idx['pairs'])==20 and sum(p['anchors'] for p in idx['pairs'])==2560,'fixed panel')
    readpin(WEIGHT);readpin(rs['runtime']);readpin(rs['main_app'])
    need(PYTHON.is_symlink() and str(PYTHON.resolve())==PYREAL,'project Python exact admitted target')
    need(str(Path(sys.executable).resolve()) in {PYREAL,rs['main_app']['path']},'actual interpreter executable pin')
    files=f['dependencies']['files'];need(len(files)==28377,'complete inherited runtime')
    digest=hashlib.sha256();total=0;empty=0
    for name,q in sorted(files.items()):
        need(q['path']==name,'runtime key');readpin(q);s=path(name).stat();total+=s.st_size;empty+=s.st_size==0
        digest.update(enc([name,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,q['sha256']]))
    need(total==819519459 and empty==752,'runtime size/empty population')
    for a in f['dependencies']['aliases']:
        p=Path(a['path']);need(p.is_symlink() and os.readlink(p)==a['link_text'] and p.resolve()==Path(a['target']),'declared runtime alias')
    aliases={a['path']:a for a in idx['input_aliases']}
    for a in aliases.values():
        p=path(a['path']);q=path(a['alias_path']);s=p.stat();t=q.stat()
        need((s.st_dev,s.st_ino,s.st_nlink)==(a['device'],a['inode'],2)==(t.st_dev,t.st_ino,t.st_nlink),'exact legacy alias pair')
        pin=dict(path=a['path'],bytes=a['bytes'],sha256=a['sha256']);readpin(pin,links=2);readpin(dict(pin,path=a['alias_path']),links=2)
    for q in idx['inputs']:readpin(q,links=2 if q['path'] in aliases else 1)
    readpin(idx['cache_manifest'])
    return dict(runtime_files=len(files),runtime_bytes=total,legitimate_empty=empty,runtime_identity_sha256=digest.hexdigest(),inputs=idx['inputs'],index=INDEX,weight=WEIGHT,freeze=OLD_FREEZE,runtime=rs['runtime'],main_app=rs['main_app'])

def collect(roots,exclude=()):
    out=set()
    for p in roots:
        p=path(p);need(p.is_relative_to(ROOT),'census root')
        if not p.exists():continue
        for q in ([p] if p.is_file() else p.rglob('*')):
            st=q.lstat();need(stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode) and st.st_nlink==1,'census aliases')
            if stat.S_ISREG(st.st_mode) and not any(q==x or x in q.parents for x in exclude):out.add(q)
    return out

def census():
    rs=json.loads(readpin(REFSPEC));readpin(rs['allocation_packet'])
    with zipfile.ZipFile(rs['allocation_packet']['path']) as z:
        m=json.loads(z.read('manifest.json'));r=m['spec']['resources'];v=json.loads(z.read('runtime/legacy-global-v4.json'))
    legacy=set()
    for d in v['domains']:
        rule=d['rule'];roots=[ROOT/n for n in rule['roots']]
        for base,pattern in rule['globs']:
            if (ROOT/base).exists():roots.extend((ROOT/base).glob(pattern))
        if rule['temp_siblings']:
            for p in list(roots):roots.extend(p.parent.glob(p.name+'.*tmp*'))
        legacy|=collect(roots,[ROOT/n for n in rule['exclude']])
    need(legacy=={ROOT/x[0] for x in v['rows']} and len(legacy)==10498,'legacy membership exact')
    groups=[legacy]
    for key,n in [('old_hq_roots',166),('ordinary_roots',41),('preserved_pixel_roots',265)]:
        g=collect([ROOT/p for p in r[key]]);need(len(g)==n,'partition '+key);groups.append(g)
    for roots,n in [([ROOT/p['path'] for p in r['closed_family_pins']],5),([ROOT/'research/hybrid-vp-startup-repair-20261023'],9),([ROOT/'research/reference-rotation-20261024'],10),([ROOT/'research/localization-uncertainty-20261024'],2),([NS],2)]:
        g=collect(roots);need(len(g)==n,'closed/new partition');groups.append(g)
    need(all(not(a&b) for i,a in enumerate(groups) for b in groups[i+1:]),'disjoint global partitions')
    files=set().union(*groups);sizes={p:p.stat().st_size for p in files}
    need(len(files)+2==11000 and not TEMP.exists(),'two retained legacy transient slots')
    need(set(NS.iterdir())=={CODE,REPORT} and CODE.stat().st_size<=131072 and REPORT.stat().st_size<=17*MIB,'two finite paths')
    fixed=v['projection']['fixed_legacy'];reserve=v['projection']['guard_reserve'];extra=sum(sizes[p] for g in groups[1:4] for p in g)+13258
    projected=dict(files=11000,npz_bytes=fixed['npz_bytes'],other_bytes=fixed['other_bytes']+extra+2*GIB+reserve+16*MIB+65536,combined_bytes=fixed['combined_bytes']+extra+2*GIB+16*MIB+65536)
    npz=sum(n for p,n in sizes.items() if p.suffix=='.npz');other=sum(sizes.values())-npz
    remaining=17*MIB+131072-sum(sizes[p] for p in groups[-1])
    need(npz==projected['npz_bytes'] and other+remaining+16*MIB+65536<=projected['other_bytes'],'whole future OTHER including stage')
    need(npz+other+remaining+16*MIB+65536<=projected['combined_bytes'],'whole future combined')
    need(projected['npz_bytes']<=12*GIB and projected['other_bytes']<=8*GIB and projected['combined_bytes']<=16*GIB,'global hard caps')
    need(all(not (ROOT/n).exists() for n in RETIRED),'only admitted ordinary retirements remain absent')
    readpin(CLOSED_V1);need(2984109+131072+17*MIB==20940973<=21009998,'entire retained original allocation')
    need(sum(sizes[p] for p in groups[-4])+16*MIB+2984109+131072+17*MIB<=2*GIB,'physical hybrid plus reference/v1/fullv2 within family reserve')
    need(shutil.disk_usage(ROOT).free>=32*GIB+remaining+16*MIB+65536,'fresh full disk reserve')
    return dict(source_history=source_history(),actual_files=len(files),membership_sha256=sha(enc(sorted(str(p.relative_to(ROOT)) for p in files))),npz_bytes=npz,other_bytes=other,projected=projected,remaining_new_bytes=remaining,temporary_path=str(TEMP),temporary_cap=16*MIB,persistent_growth_cap=65536,retired_bytes_retained=13258,closed_v1_full_future=2984109,allocation_full_future=20940973)

def host(proof):
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(proof['utc'])).total_seconds()
    need(proof['ordinaryUsageAllowed'] is True and 0<=age<=300,'fresh ordinary usage')
    raw=subprocess.check_output(['/usr/bin/vm_stat'],text=True,timeout=3);page=int(raw.split('page size of ')[1].split(' bytes')[0]);values={}
    for line in raw.splitlines()[1:]:
        if ':' in line:
            k,v=line.split(':',1);values[k]=int(v.strip().rstrip('.'))
    ram=(values['Pages free']+values['Pages inactive'])*page;need(ram>=8*GIB,'fresh available RAM')
    lines=subprocess.check_output(['/bin/ps','-ww','-axo','pid=,ppid=,pgid=,rss=,command='],text=True,timeout=3);rows={}
    for line in lines.splitlines():
        a=line.split(None,4);need(len(a)==5 and all(v.isdigit() for v in a[:4]) and int(a[0]) not in rows,'complete PS parse');rows[int(a[0])]=a
    ancestors=set();p=os.getpid()
    while p:
        need(p in rows and p not in ancestors,'launcher ancestry');ancestors.add(p);p=int(rows[p][1])
    suspects=[a for p,a in rows.items() if p not in ancestors and str(ROOT) in a[4]];need(not suspects,'no concurrent project processes')
    return dict(utc=utc(),available_ram_bytes=ram,disk_free_bytes=shutil.disk_usage(ROOT).free,process_rows=len(rows),project_suspects=suspects,ordinary_usage=proof)

def getshared():
    raw=[path(n).read_bytes() for n in SHARED]
    need(all(len(b)<=cap for b,cap in zip(raw,[16*MIB,MIB,MIB])),'shared caps')
    for n,b in zip(SHARED,raw):readpin(dict(path=n,bytes=len(b),sha256=sha(b)),16*MIB)
    return raw,json.loads(raw[0]),json.loads(raw[1])

def delta(before,after,name):
    if name==SHARED[2]:
        need(after.startswith(before),'append-only book');d=dict(codec='append',value=after[len(before):].decode('ascii'))
    else:
        a=json.loads(before);b=json.loads(after);need(isinstance(a,dict) and isinstance(b,dict),'top-level shared objects')
        d=dict(codec='json-top-level',remove=[k for k in a if k not in b],set={k:v for k,v in b.items() if k not in a or a[k]!=v})
    need(apply_delta(before,d)==after and len(enc(d))<=32768,'byteexact bounded shared delta');return d

def apply_delta(before,d):
    if d['codec']=='append':return before+d['value'].encode('ascii')
    need(d['codec']=='json-top-level','finite shared delta codec');v=json.loads(before)
    for k in d['remove']:del v[k]
    v.update(d['set']);return enc(v)

def replay_transactions():
    rows=frames(closed=True);saved=[r for r in rows if r['schema']=='shared-originals-v1']
    if not saved:return
    values=[unpack(q,16*MIB) for q in saved[0]['raw']]
    for r in rows:
        if r['schema']=='transaction-prepare-v1':
            need([dict(path=n,bytes=len(b),sha256=sha(b)) for n,b in zip(SHARED,values)]==r['before'],'historical before pins')
            values=[apply_delta(b,d) for b,d in zip(values,r['deltas'])]
            need([dict(path=n,bytes=len(b),sha256=sha(b)) for n,b in zip(SHARED,values)]==r['after'],'historical exact after pins')
    need(values==[path(n).read_bytes() for n in SHARED],'current shared equals complete journal replay')

def replace_shared(before,after,tag,nonce):
    need(not TEMP.exists() and all(len(b)<=cap for b,cap in zip(after,[16*MIB,MIB,MIB])),'stage absent and bounded')
    base=latest('freeze-v1')['shared']
    need(sum(max(0,len(a)-q['bytes']) for a,q in zip(after,base))<=65536,'positive persistent shared growth')
    changes=[delta(b,a,n) for n,b,a in zip(SHARED,before,after)]
    need(sum(len(enc(r)) for r in frames() if r['schema'].startswith('transaction-'))+len(enc(changes))<256*1024-8192,'whole transaction metadata allocation')
    identities=[]
    for n,b in zip(SHARED,before):
        p=path(n);readpin(dict(path=n,bytes=len(b),sha256=sha(b)),16*MIB);s=p.stat();identities.append([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns])
    append(dict(schema='transaction-prepare-v1',tag=tag,nonce=nonce,before=[dict(path=n,bytes=len(b),sha256=sha(b)) for n,b in zip(SHARED,before)],before_identities=identities,deltas=changes,after=[dict(path=n,bytes=len(b),sha256=sha(b)) for n,b in zip(SHARED,after)]))
    for n,old,new,identity in zip(SHARED,before,after,identities):
        p=path(n);need(p.read_bytes()==old,'shared CAS before')
        s=p.stat();need([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==identity,'raw shared identity before CAS')
        fd=os.open(TEMP,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            with os.fdopen(fd,'wb',closefd=False) as f:f.write(new);f.flush();os.fsync(fd)
            tempstat=os.fstat(fd);need(TEMP.read_bytes()==new and p.read_bytes()==old,'stage durable exact CAS')
            need((TEMP.stat().st_dev,TEMP.stat().st_ino)==(tempstat.st_dev,tempstat.st_ino),'owned stage identity')
            os.replace(TEMP,p)
            for parent in {p.parent,TEMP.parent}:
                d=os.open(parent,os.O_RDONLY);os.fsync(d);os.close(d)
            need(p.read_bytes()==new,'shared replacement readback')
        finally:os.close(fd)
    append(dict(schema='transaction-commit-v1',tag=tag,nonce=nonce,after=[actual(ROOT/n) for n in SHARED]))
    replay_transactions()

def freeze():
    need(not any(r['schema']=='freeze-v1' for r in frames()),'no duplicate freeze')
    review=latest('source-reviews-v1');need(review['source']==actual(CODE) and len(review['reviews'])==4 and {r['reviewer'] for r in review['reviews']}=={'ROOT','observability_inventory','observability_math','observability_protocol'} and all(r['decision']=='FULL_SOURCE_GO' and r['complete'] is True and r['saved_ack'] is True for r in review['reviews']),'actual saved whole source admission')
    raw,s,o=getshared();b=s['budget'];need(KEY not in s and o['active_job'] is None and s['hybrid_vp_lsd_capacity_v12']['status']=='STOPPED' and s['localization_uncertainty_v1']['status']=='STOPPED_UNTESTED' and s['localization_uncertainty_v1']['inflight'] is None,'unstarted and idle; v1 closed')
    need(b['charged_experiment_wall_seconds']==122284.44731316697 and b['committed_experiment_wall_seconds']==0 and b['limit_experiment_wall_seconds']==137100 and b['failed_distinct_families']==8,'exact current budget')
    rp=roster();c=census()
    # zlib's documented conservative bound plus Base64 and frame overhead.
    payload_bound=lambda n:4*((n+n//1000+128+2)//3)
    envelope=3*MIB+payload_bound(5*MIB)+payload_bound(2*MIB)+3*MIB+384*1024+256*1024+256*1024
    need(envelope==17005800 and envelope<=17*MIB and len(REPORT.read_bytes())<=384*1024,'complete report envelope')
    append(dict(schema='freeze-v1',utc=utc(),source=actual(CODE),roster=rp,census=c,report_envelope_bytes=envelope,shared=[actual(ROOT/n) for n in SHARED],command=command('run'),environment=ENV))

def command(mode):
    q=actual(CODE);return [str(PYTHON),'-I','-S','-B','-c',BOOT,str(CODE),str(q['bytes']),q['sha256'],mode]

def ownercheck(req):
    need(time.monotonic()<req['data_deadline'],'original data deadline')
    o=json.loads((ROOT/SHARED[1]).read_bytes());a=o['active_job']
    need(a and a['nonce']==req['nonce'] and a['source']==actual(CODE) and a['report_prefix']==req['committed_prefix'] and a['data_deadline']==req['data_deadline'],'exact owned actor admission')
    child=a['child'];need(child['pid']==os.getpid() and child['ppid']==os.getppid()==a['manager_pid'] and child['pgid']==os.getpgrp()==os.getpid() and child['role']==req['role'],'actual durable child identity')
    prefix=REPORT.read_bytes();q=req['committed_prefix'];need(len(prefix)>=q['bytes'] and sha(prefix[:q['bytes']])==q['sha256'],'committed ancestor prefix')
    frames(prefix,closed=True)
    readpin(req['source']);need(actual(REPORT)==req['report_prefix'],'actor report prefix')
    if req['role'] in ('model','audit'):need(a['native_count_cap']==req['native_count_cap'],'fixed qualified native capacity')

def shapes(logits):
    need(len(logits)==9 and all(math.isfinite(x) and -1<=x<=1 for x in logits),'finite bounded nine logits')
    m=max(logits);p=[math.exp(x-m) for x in logits];d=math.fsum(p);p=[x/d for x in p]
    mx=math.fsum(p[j]*(j%3) for j in range(9));my=math.fsum(p[j]*(j//3) for j in range(9))
    return [25*math.fsum(p[j]*v for j,v in enumerate(values)) for values in [[(j%3-mx)**2 for j in range(9)],[(j%3-mx)*(j//3-my) for j in range(9)],[(j//3-my)**2 for j in range(9)]]]

def project(H,p):
    x,y=p;w=H[2][0]*x+H[2][1]*y+H[2][2];need(math.isfinite(w) and abs(w)>1e-12,'defined projective denominator')
    v=[(H[k][0]*x+H[k][1]*y+H[k][2])/w for k in range(2)]
    J=[[(H[k][j]-v[k]*H[2][j])/w for j in range(2)] for k in range(2)]
    need(all(math.isfinite(x) for x in v+sum(J,[])),'finite projection/J');return v,J

def propagate(A,B,J):
    a,b,c=A;u,v=J[0];w,z=J[1]
    return [B[0]+u*u*a+2*u*v*b+v*v*c,B[1]+u*w*a+(u*z+v*w)*b+v*z*c,B[2]+w*w*a+2*w*z*b+z*z*c]

def quadratic(S,r):
    a,b,c=S;d=a*c-b*b;need(all(math.isfinite(v) for v in S+r) and a>0 and math.isfinite(d) and d>0,'finite SPD covariance')
    q=(c*r[0]*r[0]-2*b*r[0]*r[1]+a*r[1]*r[1])/d
    ld=math.log(d);need(math.isfinite(q) and q>=0 and math.isfinite(ld),'finite nonnegative Q/logdet');return q,ld

def fixtures():
    tests=[]
    def check(name,condition):need(condition,name);tests.append(name)
    H=shapes([0.]*9);check('uniform shape',abs(H[0]-50/3)<1e-12 and abs(H[1])<1e-12 and abs(H[2]-50/3)<1e-12)
    for z in ([float('nan')]*9,[2.]*9,[0.]*8):
        try:shapes(z)
        except RuntimeError:tests.append('invalid logits rejected')
        else:raise RuntimeError('invalid logits accepted')
    q,ld=quadratic([2.,0.,8.],[2.,4.]);check('known Q/logdet',q==4 and abs(ld-math.log(16))<1e-15)
    alpha=2.;check('known NLL',abs(.5*(2*math.log(alpha)+ld+q/alpha+2*math.log(2*math.pi))-(math.log(2)+math.log(4)+1+math.log(2*math.pi)))<1e-14)
    for S in ([0.,0.,1.],[1.,2.,1.],[1e200,0.,1e200]):
        try:quadratic(S,[1.,2.])
        except RuntimeError:tests.append('nonSPD rejected')
        else:raise RuntimeError('nonSPD accepted')
    T=[[1.2,.1,3.],[-.2,.9,7.],[.0003,-.0002,1.]];p=[31.,47.];v,J=project(T,p)
    check('identity project',project([[1,0,0],[0,1,0],[0,0,1]],p)==(p,[[1.,0.],[0.,1.]]))
    for j in range(2):
        pp=p.copy();pm=p.copy();pp[j]+=1e-3;pm[j]-=1e-3;vp,_=project(T,pp);vm,_=project(T,pm)
        check('projective Jacobian',max(abs((vp[k]-vm[k])/.002-J[k][j]) for k in range(2))<1e-8)
    check('constant rotate propagation',propagate([1,0,1],[1,0,1],[[0,-1],[1,0]])==[2,0,2])
    check('constant scale propagation',propagate([1,0,1],[1,0,1],[[2,0],[0,3]])==[5,0,10])
    check('scalar calibration',math.fsum([2,4,6])/(2*3)==2)
    a=list(range(128));b=a.copy();random.Random(20261024).shuffle(a);random.Random(20261024).shuffle(b)
    check('fixed shuffle denominator',a==b and sorted(a)==list(range(128)) and a!=list(range(128)))
    first=enc({'old':7,'x':1});second=enc({'new':[True],'x':2});change=delta(first,second,SHARED[0]);check('typed JSON delta exact',apply_delta(first,change)==second)
    change=delta(b'x',b'xy',SHARED[2]);check('book delta exact',apply_delta(b'x',change)==b'xy')
    try:frames(b'<!-- RECORD -->\n```json\n{}\n```\n')
    except (RuntimeError,KeyError):tests.append('malformed frame rejected')
    else:raise RuntimeError('malformed frame accepted')
    initial=b'<!-- RECORD -->\n```json\n{"schema":"uncertainty-prospective-v1"}\n```\n'
    for tail in [b'\n<!-- RECORD -->\n```json\n{',b'garbage',b'\n<!-- RECORD -->\n```json\n{"schema":"freeze-v1"}\n```\n\n<!-- RECORD -->\n```json\n{"schema":"freeze-v1"}\n```\n',b'\n<!-- RECORD -->\n```json\n{"nonce":"x","schema":"transaction-commit-v1","tag":"x"}\n```\n',b'\n<!-- RECORD -->\n```json\n{"nonce":"x","schema":"transaction-prepare-v1","tag":"x"}\n```\n']:
        try:frames(initial+tail,closed=True)
        except (RuntimeError,KeyError,ValueError):tests.append('duplicate/missing/partial journal rejected')
        else:raise RuntimeError('invalid journal accepted')
    for value in [0.,-1.,float('nan'),float('inf')]:
        try:need(math.isfinite(value) and value>0,'positive scale')
        except RuntimeError:tests.append('invalid scale rejected')
        else:raise RuntimeError('invalid scale accepted')
    for value in [True,False,'12',None]:
        try:need(type(value) in (int,float) and math.isfinite(value),'numeric wire')
        except (RuntimeError,TypeError):tests.append('invalid numeric wire rejected')
        else:raise RuntimeError('invalid numeric wire accepted')
    for n,c in [(1,512),(256,512),(257,1024),(768,1024),(769,2048),(1792,2048)]:check('fixed native capacity',native_capacity(n)==c)
    for n in [0,1793,4097,True]:
        try:native_capacity(n)
        except RuntimeError:tests.append('invalid/unaffordable count rejected')
        else:raise RuntimeError('invalid native capacity accepted')
    example=[['a','x.py'],['b','x.py'],['c','y.py']];check('origin index transport exact',compact_origins(example,{'x.py':{},'y.py':{}})==[['a',0],['b',0],['c',1]])
    def rejects(name,fn):
        try:fn()
        except (RuntimeError,KeyError,ValueError,TypeError):tests.append(name)
        else:raise RuntimeError('invalid fixture accepted:'+name)
    base=dict(status='COUNT_OBSERVED',forward_calls=0,homography_values_read=False,dataset_arrays_decoded=False,snapshot_shape_controls=['real_numpy_scalar_shape','real_numpy_strided_C_order','real_numpy_NaN_scalar_bits'],actual_executable=PYREAL,state_sha256='c0e9533a7747ea4288fd3576f49bc779ab91066c2a5246591073463f61893675',diagnostic=dict(count=600,repeated_count=600,uint32_bytes=4,abi='uint32_t _dyld_image_count(void)',old_512_predicate=False))
    check('strict count metadata positive',validate_count_reply(base,{PYREAL})['count']==600)
    for field,value in [('forward_calls',False),('actual_executable','/unknown/python'),('homography_values_read',0)]:
        bad=json.loads(enc(base));bad[field]=value;rejects('strict count scope/runtime rejects',lambda bad=bad:validate_count_reply(bad,{PYREAL}))
    for field,value in [('count',True),('repeated_count',600.),('uint32_bytes',4.),('abi','wrong'),('old_512_predicate',0),('old_512_predicate',True)]:
        bad=json.loads(enc(base));bad['diagnostic'][field]=value;rejects('strict count ABI/types rejects',lambda bad=bad:validate_count_reply(bad,{PYREAL}))
    row=dict(path='',scope='TRUSTED_OS_OR_FIXED_CPYTHON');pad=512-len(enc(row));row['path']='x'*pad
    check('native EXACT512 encoded bytes',len(native_row_wire(row))==512)
    rejects('native513 rejects',lambda:native_row_wire(dict(row,path=row['path']+'x')))
    rejects('native unicode escape bytes rejects',lambda:native_row_wire(dict(row,path='é'*pad)))
    check('evidence exact5MiB with framecharge',evidence_charge(5*MIB-4096,0)==5*MIB)
    rejects('evidence one-byte overflow rejects',lambda:evidence_charge(5*MIB-4096,1))
    check('origin exact256KiB wire',len(origin_wire(b'x'*(256*1024)))==256*1024)
    rejects('origin256KiBplus1 rejects',lambda:origin_wire(b'x'*(256*1024+1)))
    def descriptor(values,shape,dtype):
        raw=b''.join(struct.pack('<f' if dtype=='<f4' else '<q',v) for v in values)
        return dict(dtype=dtype,shape=list(shape),bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode())
    inp0=descriptor([-0.,.5]*128,(128,2),'<f4');inp1=descriptor([4.,5.]*128,(128,2),'<f4')
    out0=descriptor([1.25,2.5]*128,(128,2),'<f4');out1=descriptor([7.25,8.5]*128,(128,2),'<f4')
    target=[0.,.125,.25,.375,.5,.625,.75,.875,1.];source=[-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.]
    common=dict(group='CODEC_FIXTURE_ONLY',inputs=[inp0,inp1],ids=descriptor(list(range(128)),(128,),'<i8'),hw=[[24,24],[24,24]],array_hashes={})
    control=dict(common,pair=0,kind='plain',forward_ordinal=1,outputs=[out0,out1]);chunks=[dict(common,pair=i,kind='primary',forward_ordinal=i+2,outputs=[out0,out1],maps=[descriptor(target*128,(128,1,3,3),'<f4'),descriptor(source*128,(128,1,3,3),'<f4')]) for i in range(2)]
    a=decode_pair(chunks[0]);check('binary independent known endpoints/maps',a['out0'][0]==[1.25,2.5] and a['out1'][0]==[7.25,8.5] and a['logits0'][0]==source and a['logits1'][0]==target and math.copysign(1,a['input0'][0][0])<0)
    metadata=dict(hook_parity=True,forwards=3);expected=dict(metadata,pairs=[decode_pair(v) for v in chunks]);manifest=dict(metadata=metadata,expected_pairs=2,pair_fragments=[dict(pair=i,sha256=sha(enc(v))) for i,v in enumerate(chunks)],plain_sha256=sha(enc(control)),logical_model_sha256=sha(enc(expected)))
    raw,_=assemble_model(manifest,chunks,control,expected_pairs=2);check('full binary manifest roundtrip',raw==enc(expected))
    rejects('fragment missing rejects',lambda:assemble_model(manifest,chunks[:1],control,expected_pairs=2))
    rejects('fragment order rejects',lambda:assemble_model(manifest,list(reversed(chunks)),control,expected_pairs=2))
    rejects('fragment duplicate rejects',lambda:assemble_model(manifest,[chunks[0],chunks[0]],control,expected_pairs=2))
    bad=json.loads(enc(chunks));bad[0]['outputs'][0]['base64']='AAAA'+bad[0]['outputs'][0]['base64'][4:]
    rejects('fragment exact SHA tamper rejects',lambda:assemble_model(manifest,bad,control,expected_pairs=2))
    badmanifest=dict(manifest,logical_model_sha256='0'*64);rejects('logical model SHA rejects',lambda:assemble_model(badmanifest,chunks,control,expected_pairs=2))
    bits=bytes.fromhex('0100c07f');nan=dict(dtype='<f4',shape=[1],bytes=4,sha256=sha(bits),base64=base64.b64encode(bits).decode())
    check('NaN bits retainable before validation',math.isnan(tensor_flat(nan,(1,),'<f4')[0]) and base64.b64decode(nan['base64'])==bits)
    badtensor=dict(inp0,sha256='0'*64);rejects('tensor raw-bit SHA rejects',lambda:tensor_flat(badtensor,(128,2),'<f4'))
    route=[dict(schema='native-before-v2',nonce='fixture'),dict(schema='model-plain-v2',nonce='fixture')]+[dict(schema='model-pair-v2',nonce='fixture',pair_index=i) for i in range(2)]+[dict(schema='model-raw-v1',nonce='fixture')]
    check('retained core readable before after-native',validate_model_order(route,'fixture',expected_pairs=2,require_after=False))
    rejects('unqualified native-after cannot score',lambda:validate_model_order(route,'fixture',expected_pairs=2))
    check('qualified metadata route',validate_model_order(route+[dict(schema='native-after-v2',nonce='fixture')],'fixture',expected_pairs=2))
    rejects('rejected native cannot score',lambda:validate_model_order(route+[dict(schema='native-rejected-v2',nonce='fixture')],'fixture',expected_pairs=2))
    return tests

def npy_rows(body):
    need(body[:6]==b'\x93NUMPY','NPY magic');major=body[6];need(major in (1,2),'NPY format')
    k=2 if major==1 else 4;n=int.from_bytes(body[8:8+k],'little');header=ast.literal_eval(body[8+k:8+k+n].decode('latin1'))
    need(header['fortran_order'] is False and header['descr'] in ('<f4','<i8'),'NPY scalar type/order');shape=header['shape'];need(shape in ((128,2),(128,)),'NPY fixed shape')
    code='f' if header['descr']=='<f4' else 'q';payload=body[8+k+n:];need(len(payload)==math.prod(shape)*struct.calcsize(code),'NPY exact payload')
    values=[x[0] for x in struct.iter_unpack('<'+code,payload)]
    return [values[j:j+2] for j in range(0,len(values),2)] if len(shape)==2 else values

def wire_panel(model,idx):
    need(model['hook_parity'] is True and type(model['forwards']) is int and model['forwards']==21 and len(model['pairs'])==20,'sealed model coverage')
    for item,out in zip(idx['pairs'],model['pairs']):
        need(type(out['pair']) is int and out['pair']==item['pair_index'] and out['group']==item['source_group'] and out['hw']==item['original_hw'],'ordered wire authority')
        need(isinstance(out['ids'],list) and len(out['ids'])==128 and all(type(x) is int for x in out['ids']) and len(set(out['ids']))==128,'strict anchor identities')
        for field,width in [('out0',2),('out1',2),('input0',2),('input1',2),('logits0',9),('logits1',9)]:
            need(isinstance(out[field],list) and len(out[field])==128 and all(isinstance(row,list) and len(row)==width and all(type(x) in (int,float) and math.isfinite(x) for x in row) for row in out[field]),'strict finite raw wire')
        with zipfile.ZipFile(io.BytesIO(readpin(item['npz']))) as z:
            for member,field in [('p0.npy','input0'),('v0.npy','input1'),('anchor_indices.npy','ids')]:
                info=z.getinfo(member);need(info.file_size<=65536,'bounded fixed-anchor member');need(npy_rows(z.read(member))==out[field],'independent original anchor readback')

def arrayhash(np,a):
    a=np.ascontiguousarray(a);return sha(str((a.shape,a.dtype)).encode()+a.tobytes())

def native_count():
    import ctypes
    lib=ctypes.CDLL(None);count=lib._dyld_image_count;count.argtypes=[];count.restype=ctypes.c_uint32
    n=count();again=count();return dict(count=n,repeated_count=again,uint32_bytes=ctypes.sizeof(ctypes.c_uint32),old_512_predicate=bool(0<n<=512),abi='uint32_t _dyld_image_count(void)')


def native_capacity(n):
    need(type(n) is int and 1<=n<=4096,'native count outside finite qualification')
    cap=1<<(max(512,n+256)-1).bit_length();need(cap<=2048,'UNAFFORDABLE_NATIVE_WIRE; no clamp/reselection');return cap


MODEL_PARTS={'native-before-v2','model-plain-v2','model-pair-v2','model-raw-v1','native-after-v2','native-rejected-v2'}


def evidence_charge(previous,raw_size):
    need(type(previous) is int and type(raw_size) is int and previous>=0 and raw_size>=0,'typed evidence sizes')
    total=previous+raw_size+4096;need(total<=5*MIB,'cumulative model evidence including per-member framing');return total


def native_row_wire(row):
    raw=enc(row);need(len(raw)<=512,'native WHOLE encoded row exceeds512');return raw


def origin_wire(raw):
    need(isinstance(raw,bytes) and len(raw)<=256*1024,'complete Python origin wire cap');return raw


def tensor_snapshot(np,value):
    raw=value.tobytes(order='C');need(len(raw)<=128*1024,'bounded returned tensor snapshot')
    return dict(dtype=value.dtype.str,shape=list(value.shape),bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode())


def numpy_snapshot_fixtures(np):
    scalar=np.asarray(-0.0,dtype=np.float32);a=tensor_snapshot(np,scalar)
    need(a['shape']==[] and a['dtype']==scalar.dtype.str and base64.b64decode(a['base64'])==struct.pack('<f',-0.) and math.copysign(1,tensor_flat(a,(),'<f4')[0])<0,'REAL NumPy scalar original shape/negative-zero bits')
    view=np.arange(12,dtype=np.float32).reshape(3,4)[:,::2];b=tensor_snapshot(np,view)
    need(b['shape']==[3,2] and b['dtype']==view.dtype.str and base64.b64decode(b['base64'])==struct.pack('<6f',0,2,4,6,8,10) and tensor_flat(b,(3,2),'<f4')==[0,2,4,6,8,10],'REAL NumPy noncontiguous original shape/C-order bytes')
    nan=np.asarray(np.nan,dtype=np.float32);c=tensor_snapshot(np,nan)
    need(c['shape']==[] and base64.b64decode(c['base64'])==nan.tobytes(order='C') and math.isnan(tensor_flat(c,(),'<f4')[0]),'REAL NumPy NaN scalar shape/raw-bit retention')
    return ['real_numpy_scalar_shape','real_numpy_strided_C_order','real_numpy_NaN_scalar_bits']


def source_history():
    admission=latest('applied-v2-source-storage-and-preserved-82514-v1')
    need(admission['status']=='APPLIED_SOURCE_STORAGE_ONLY' and admission['aggregate_source_versions_cap']==393216 and {r['reviewer'] for r in admission['actual_common_record_acks']}=={'ROOT','observability_inventory','observability_math','observability_protocol'} and all(r['actual'] is True for r in admission['actual_common_record_acks']),'actual applied bounded source-storage admission')
    current=actual(CODE);need(current['bytes']<=131072,'current source body cap');total=current['bytes'];archives=[]
    for row in frames():
        value=row.get('source')
        if isinstance(value,dict) and 'zlib_base64' in value:
            raw=unpack(value,131072);need(0<len(raw)<=131072,'preserved source body cap');total+=len(raw);archives.append(dict(schema=row['schema'],bytes=len(raw),sha256=sha(raw)))
    need(total<=393216,'all actual decoded source versions cap; no dedupcredit');return dict(total_bytes=total,cap=393216,current=current,archives=archives)


def tensor_flat(desc,shape,dtype):
    need(set(desc)=={'dtype','shape','bytes','sha256','base64'} and desc['dtype']==dtype and desc['shape']==list(shape) and all(type(n) is int for n in desc['shape']),'exact binary tensor descriptor')
    width=4 if dtype=='<f4' else 8 if dtype=='<i8' else None;need(width is not None,'fixed tensor dtype')
    need(type(desc['bytes']) is int and desc['bytes']==math.prod(shape)*width<=128*1024 and isinstance(desc['base64'],str) and len(desc['base64'])<=4*((desc['bytes']+2)//3),'typed tensor byte bounds')
    raw=base64.b64decode(desc['base64'],validate=True);need(len(raw)==desc['bytes'] and sha(raw)==desc['sha256'],'exact binary tensor hash')
    return [v[0] for v in struct.iter_unpack('<f' if dtype=='<f4' else '<q',raw)]


def decode_pair(snapshot):
    need(snapshot['kind']=='primary' and type(snapshot['pair']) is int and len(snapshot['outputs'])==2 and len(snapshot['maps'])==2 and len(snapshot['inputs'])==2,'fixed pair snapshot structure')
    a,b=[tensor_flat(d,(128,2),'<f4') for d in snapshot['inputs']];u,v=[tensor_flat(d,(128,2),'<f4') for d in snapshot['outputs']];target,source=[tensor_flat(d,(128,1,3,3),'<f4') for d in snapshot['maps']]
    chunk=lambda value,k:[value[j:j+k] for j in range(0,len(value),k)]
    return dict(pair=snapshot['pair'],group=snapshot['group'],ids=tensor_flat(snapshot['ids'],(128,),'<i8'),input0=chunk(a,2),input1=chunk(b,2),out0=chunk(u,2),out1=chunk(v,2),logits0=chunk(source,9),logits1=chunk(target,9),hw=snapshot['hw'],array_hashes=snapshot['array_hashes'])


def assemble_model(manifest,pair_snapshots,plain,expected_pairs=20):
    need(type(manifest['expected_pairs']) is int and manifest['expected_pairs']==expected_pairs and len(manifest['pair_fragments'])==len(pair_snapshots)==expected_pairs,'complete manifest population')
    need(plain['kind']=='plain' and type(plain['pair']) is int and plain['pair']==0 and type(plain['forward_ordinal']) is int and plain['forward_ordinal']==1 and sha(enc(plain))==manifest['plain_sha256'],'authenticated plain/control snapshot')
    pairs=[]
    for i,(reference,snapshot) in enumerate(zip(manifest['pair_fragments'],pair_snapshots)):
        need(type(reference['pair']) is int and type(snapshot['pair']) is int and reference['pair']==snapshot['pair']==i and type(snapshot['forward_ordinal']) is int and snapshot['forward_ordinal']==i+2 and sha(enc(snapshot))==reference['sha256'],'exact ordered nonduplicate pair fragment')
        pairs.append(decode_pair(snapshot))
    first=pair_snapshots[0];need(len(plain['outputs'])==2,'plain returned outputs')
    for a,b in zip(plain['outputs'],first['outputs']):
        need(a==b,'plain/hook dtype shape and bytes descriptor parity')
    metadata=manifest['metadata'];need(metadata['hook_parity'] is True and type(metadata['forwards']) is int and metadata['forwards']==expected_pairs+1,'authenticated forward denominator')
    model=dict(metadata,pairs=pairs);raw=enc(model);need(sha(raw)==manifest['logical_model_sha256'],'exact reconstructed logical model hash');return raw,model


def validate_model_order(rows,nonce,expected_pairs=20,require_after=True):
    parts=[r for r in rows if r['schema'] in MODEL_PARTS];need(all(r['nonce']==nonce for r in parts),'one exact model-fragment nonce')
    need(not any(r['schema']=='native-rejected-v2' for r in parts),'rejected native route cannot score')
    pairs=[r for r in parts if r['schema']=='model-pair-v2'];need(len(pairs)==expected_pairs and [r['pair_index'] for r in pairs]==list(range(expected_pairs)) and all(type(r['pair_index']) is int for r in pairs),'strict ordered complete pair route')
    one=lambda schema:[r for r in parts if r['schema']==schema]
    before=one('native-before-v2');plain=one('model-plain-v2');core=one('model-raw-v1');after=one('native-after-v2')
    need(len(before)==len(plain)==len(core)==1 and len(after)==(1 if require_after else len(after)) and len(after)<=1,'complete required model route')
    ordered=before+plain+pairs+core+after;positions=[rows.index(r) for r in ordered];need(positions==sorted(positions) and len(set(positions))==len(positions),'actual durable native/plain/pair/manifest order')
    return True


def read_model_core():
    rows=frames();core=latest('model-raw-v1');validate_model_order(rows,core['nonce'],require_after=False);manifest=json.loads(unpack(core['payload'],512*1024));parts=[r for r in rows if r['schema']=='model-pair-v2'];plain=latest('model-plain-v2')
    need(len(parts)==20 and all(r['nonce']==core['nonce'] for r in parts+[plain]),'one exact fragment nonce/population')
    positions=[rows.index(plain)]+[rows.index(r) for r in parts]+[rows.index(core)];need(positions==sorted(positions) and len(set(positions))==22,'actual durable fragment/manifest order')
    snapshots=[json.loads(unpack(r['payload'],256*1024)) for r in parts];control=json.loads(unpack(plain['payload'],256*1024))
    return assemble_model(manifest,snapshots,control)


def validate_count_reply(value,allowed_executables):
    need(isinstance(value,dict) and value['status']=='COUNT_OBSERVED' and type(value['forward_calls']) is int and value['forward_calls']==0 and value['homography_values_read'] is False and value['dataset_arrays_decoded'] is False,'strict count-only result scope')
    need(isinstance(value['actual_executable'],str) and value['actual_executable'] in allowed_executables and value['state_sha256']=='c0e9533a7747ea4288fd3576f49bc779ab91066c2a5246591073463f61893675','actual admitted count runtime/state')
    need(value['snapshot_shape_controls']==['real_numpy_scalar_shape','real_numpy_strided_C_order','real_numpy_NaN_scalar_bits'],'actual NumPy snapshot controls completed before count')
    d=value['diagnostic'];need(isinstance(d,dict) and type(d['count']) is int and type(d['repeated_count']) is int and 0<=d['count']<=4294967295 and d['count']==d['repeated_count'],'strict stable uint32 count')
    need(type(d['uint32_bytes']) is int and d['uint32_bytes']==4 and d['abi']=='uint32_t _dyld_image_count(void)' and type(d['old_512_predicate']) is bool and d['old_512_predicate']==bool(0<d['count']<=512),'exact count ABI/old-predicate witness')
    return d


def model_part(req,schema,value):
    need(schema in MODEL_PARTS,'finite model fragment kind');raw=enc(value);prior=[r for r in frames() if r['schema'] in MODEL_PARTS]
    need(all(r['nonce']==req['nonce'] and r['charge_bytes']==r['payload']['bytes']+4096 for r in prior),'exact existing fragment charges')
    evidence_charge(sum(r['charge_bytes'] for r in prior),len(raw))
    if schema=='model-pair-v2':need(not any(r.get('pair_index')==value['pair'] for r in prior if r['schema']==schema),'no duplicate pair publication')
    before=actual(REPORT);need(before==req['report_prefix'],'model owns current evidence prefix')
    row=dict(schema=schema,nonce=req['nonce'],payload=packed(raw),charge_bytes=len(raw)+4096)
    if schema=='model-pair-v2':row['pair_index']=value['pair']
    prefix=append(row,before);req['report_prefix']=prefix
    return dict(schema=schema,bytes=len(raw),sha256=sha(raw),report=prefix)




def native_images(f,req,phase):
    import ctypes
    lib=ctypes.CDLL(None);count=lib._dyld_image_count;count.argtypes=[];count.restype=ctypes.c_uint32
    name=lib._dyld_get_image_name;name.argtypes=[ctypes.c_uint32];name.restype=ctypes.c_char_p
    n=count();cap=req['native_count_cap'];facts=dict(phase=phase,observed_count=n,capacity=cap,old_512_predicate=bool(0<n<=512),names=[],collection_complete=False,offending_index=None)
    try:
        need(0<n<=cap<=2048,'actual native count outside qualified capacity; n='+str(n))
        for i in range(n):
            facts['offending_index']=i;raw=name(i);need(raw is not None,'null actual native image name')
            facts['names'].append(os.fsdecode(raw));need(len(enc(facts))<=MIB+4096,'native name collection resource bound')
        facts['collection_complete']=True;facts['offending_index']=None
        rows=[];files=f['dependencies']['files'];aliases={x['path']:x for x in f['dependencies']['aliases']}
        for i,observed in enumerate(facts['names']):
            facts['offending_index']=i;p=Path(observed);need(p.is_absolute(),'absolute native origin')
            if p.is_relative_to(ROOT):
                if str(p) in aliases:
                    a=aliases[str(p)];need(p.is_symlink() and os.readlink(p)==a['link_text'] and p.resolve()==Path(a['target']),'native declared alias');p=Path(a['target'])
                rel=p.relative_to(ROOT).as_posix();need(rel in files,'unknown ROOT native origin:'+observed);readpin(files[rel]);row=dict(path=str(p),scope='HASHED_ROOT',sha256=files[rel]['sha256'])
                if observed!=str(p):row['observed_path']=observed
            else:
                allowed=any(p.is_relative_to(Path(prefix)) for prefix in ['/System/Library','/usr/lib','/Library/Apple/System/Library','/Library/Frameworks/Python.framework/Versions/3.11'])
                need(allowed,'unknown external native origin:'+observed);row=dict(path=observed,scope='TRUSTED_OS_OR_FIXED_CPYTHON')
            native_row_wire(row);rows.append(row)
        observed_after=count();facts['observed_after']=observed_after;need(observed_after==n,'native snapshot changed; before='+str(n)+' after='+str(observed_after))
        payload=dict(schema='sampled-native-roster-v2',phase=phase,status='PASS',capacity=cap,count=n,count_after=observed_after,complete=True,continuous_history_claim=False,rows=rows)
        need(len(enc(payload))<=cap*512+4096,'native complete container allowance');return model_part(req,'native-'+phase+'-v2',payload)
    except BaseException as e:
        facts.update(error=type(e).__name__+':'+str(e),status='REJECTED',continuous_history_claim=False)
        model_part(req,'native-rejected-v2',facts);raise


def compact_origins(origins,files):
    ordered=sorted(files);index={p:i for i,p in enumerate(ordered)};compact=[[name,index[p]] for name,p in origins]
    need([[name,ordered[i]] for name,i in compact]==origins,'lossless origin-index roundtrip');origin_wire(enc(compact));return compact


def native_model_evidence(nonce):
    rows=frames();validate_model_order(rows,nonce);allparts=[r for r in rows if r['schema'] in MODEL_PARTS]
    need(all(r['nonce']==nonce and r['charge_bytes']==r['payload']['bytes']+4096 for r in allparts) and sum(r['charge_bytes'] for r in allparts)<=5*MIB,'complete cumulative native/model evidence/framing')
    need(not any(r['schema']=='native-rejected-v2' for r in allparts),'native inspection must be qualified before scoring')
    admission=latest('native-capacity-admission-v2');need(admission['nonce']==nonce and admission['capacity']==native_capacity(admission['observed_count']),'same predeclared COUNT capacity')
    native=[]
    for schema in ('native-before-v2','native-after-v2'):
        part=latest(schema);v=json.loads(unpack(part['payload'],2*MIB));need(v['status']=='PASS' and v['complete'] is True and type(v['count']) is int and 0<v['count']==v['count_after']<=v['capacity']==admission['capacity']<=2048 and len(v['rows'])==v['count'],'complete qualified sampled native roster')
        for row in v['rows']:native_row_wire(row)
        native.append(part)
    need(rows.index(native[0])<rows.index(latest('model-plain-v2'))<rows.index(latest('model-raw-v1'))<rows.index(native[1]),'native/core durable ordering')
    raw,core=read_model_core();need(core['native_before_sha256']==native[0]['payload']['sha256'],'core binds actual before-native evidence');files=json.loads(readpin(OLD_FREEZE))['dependencies']['files'];ordered=sorted(files);origin_wire(enc(core['loaded_origin_indices']))
    for name,index in core['loaded_origin_indices']:
        need(isinstance(name,str) and type(index) is int and 0<=index<len(ordered),'typed origin index');need(ordered[index] in files,'pinned expanded origin')
    return True





def model_actor(req):
    ownercheck(req);tests=fixtures();old=old_leaves();f=json.loads(readpin(OLD_FREEZE));gate=old.FrozenGate(f['dependencies']['files']);sys.meta_path.insert(0,gate)
    for p in [old.SITE,ROOT/'vendor/XRefine',ROOT/'vendor/XRefine/submodules/glue_factory']:sys.path.insert(0,str(p))
    import platform
    architecture=os.uname().machine;platform._Processor.get=staticmethod(lambda:architecture)
    def no_spawn(event,args):
        if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.system'}:raise RuntimeError('actor spawning forbidden')
    sys.addaudithook(no_spawn)
    import numpy as np
    snapshot_shape_controls=numpy_snapshot_fixtures(np);tests.extend(snapshot_shape_controls)
    import torch
    import cv2
    need(not torch.cuda.is_available(),'CPU before constructor');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);cv2.setNumThreads(0);need(cv2.getNumThreads()==1,'CV serial')
    import model as author
    need(Path(author.__file__)==ROOT/'vendor/XRefine/model.py','exact author module')
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))]):
        ckpt=torch.load(io.BytesIO(readpin(WEIGHT)),map_location='cpu',weights_only=True)
    net=author.XRefine('small',False,True).cpu();net.net.load_state_dict(ckpt['model'],strict=True);net.eval()
    for p in net.parameters():p.requires_grad_(False)
    need(all(not m.training for m in net.modules()) and all(p.device.type=='cpu' for p in net.parameters()),'CPU frozen eval')
    need(net.net.spatial_argmax.temperature==1. and net.net.spatial_argmax.eps==1e-6,'unchanged decoder')
    h=hashlib.sha256()
    for name,tensor in sorted(net.state_dict().items()):
        value=tensor.detach().cpu().contiguous();head=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode();body=value.numpy().tobytes();h.update(len(head).to_bytes(8,'little'));h.update(head);h.update(len(body).to_bytes(8,'little'));h.update(body)
    need(h.hexdigest()=='c0e9533a7747ea4288fd3576f49bc779ab91066c2a5246591073463f61893675','accepted matched STAR state')
    if req['role']=='count':
        diagnostic=native_count();return dict(status='COUNT_OBSERVED',diagnostic=diagnostic,state_sha256=h.hexdigest(),forward_calls=0,homography_values_read=False,dataset_arrays_decoded=False,snapshot_shape_controls=snapshot_shape_controls,actual_executable=sys.executable,report=actual(REPORT))
    native_before=native_images(f,req,'before')
    idx=json.loads(readpin(INDEX));manifest=json.loads(readpin(idx['cache_manifest']));pairs=[];references=[];timings=[];forwards=0;plain_reference=None
    for item in idx['pairs']:
        need(time.monotonic()<req['data_deadline'],'model data bound');i=item['pair_index'];record=manifest['pairs'][i]
        need((record['sequence'],record['source'],record['target'])==(item['sequence'],item['source'],item['target']),'manifest join')
        names=['image0_rgb','image1_rgb','p0','v0','all_p0','all_v0','anchor_indices']
        with np.load(io.BytesIO(readpin(item['npz'])),allow_pickle=False) as z:arrays={k:z[k].copy() for k in names}
        hashes={k:arrayhash(np,a) for k,a in arrays.items()};need(all(hashes[k]==record['inference_array_sha256'][k] for k in names),'required array hashes')
        a,b=arrays['p0'],arrays['v0'];ids=arrays['anchor_indices'];n=len(arrays['all_p0']);expected=np.arange(n,dtype=np.int64) if n<=128 else np.floor(np.linspace(0,n-1,128)).astype(np.int64)
        need(a.shape==b.shape==(128,2) and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'fixed finite coordinates')
        need(np.array_equal(ids,expected) and np.array_equal(a,arrays['all_p0'][ids]) and np.array_equal(b,arrays['all_v0'][ids]),'fixed uniform anchor identity')
        images=[]
        for side in (0,1):
            rgb=arrays['image%d_rgb'%side];need(rgb.dtype==np.float32 and rgb.shape==(*item['original_hw'][side],3) and np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1,'native RGB')
            images.append(torch.from_numpy(rgb.transpose(2,0,1).copy()))
        pa,pb=torch.from_numpy(a),torch.from_numpy(b)
        common=dict(pair=i,group=item['source_group'],inputs=[tensor_snapshot(np,a),tensor_snapshot(np,b)],ids=tensor_snapshot(np,ids),hw=item['original_hw'],array_hashes=hashes)
        if i==0:
            plain=net(pa,pb,*images);forwards+=1
            control=dict(common,kind='plain',forward_ordinal=forwards,outputs=[tensor_snapshot(np,x.detach().cpu().numpy()) for x in plain])
            plain_reference=model_part(req,'model-plain-v2',control)
        maps=[];hook=net.net.score_map_head.register_forward_hook(lambda m,args,out:maps.append(out.detach().cpu().numpy().copy()))
        t=time.monotonic()
        try:out=net(pa,pb,*images);forwards+=1
        finally:hook.remove()
        timings.append(time.monotonic()-t)
        values=[x.detach().cpu().numpy().copy() for x in out]
        snapshot=dict(common,kind='primary',forward_ordinal=forwards,outputs=[tensor_snapshot(np,x) for x in values],maps=[tensor_snapshot(np,x) for x in maps])
        reference=model_part(req,'model-pair-v2',snapshot)
        # Every returned control/output/map is durable BEFORE rejecting checks.
        need(len(maps)==2 and all(x.shape==(128,1,3,3) and x.dtype==np.float32 and np.isfinite(x).all() and (np.abs(x)<=1).all() for x in maps),'target/source raw map calls')
        need(len(values)==2 and all(x.shape==(128,2) and x.dtype==np.float32 and np.isfinite(x).all() for x in values),'finite author coordinates')
        if i==0:need(control['outputs']==snapshot['outputs'],'hook/plain byte exact descriptors')
        pair=decode_pair(snapshot)
        expected_pair=dict(pair=i,group=item['source_group'],ids=ids.tolist(),input0=a.tolist(),input1=b.tolist(),out0=values[0].tolist(),out1=values[1].tolist(),logits0=maps[1].reshape(128,9).tolist(),logits1=maps[0].reshape(128,9).tolist(),hw=item['original_hw'],array_hashes=hashes)
        need(enc(pair)==enc(expected_pair),'REAL NumPy/binary-codec exact equivalence for every pair')
        pairs.append(pair);references.append(dict(pair=i,sha256=reference['sha256']))
    need(len(pairs)==20 and forwards==21 and plain_reference is not None,'all requested forwards')
    final_state=hashlib.sha256()
    for name,tensor in sorted(net.state_dict().items()):
        value=tensor.detach().cpu().contiguous();head=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode();body=value.numpy().tobytes();final_state.update(len(head).to_bytes(8,'little'));final_state.update(head);final_state.update(len(body).to_bytes(8,'little'));final_state.update(body)
    need(final_state.hexdigest()==h.hexdigest(),'final model state unchanged')
    origins=[]
    for name,m in list(sys.modules.items()):
        origin=getattr(m,'__file__',None)
        if origin and Path(origin).is_relative_to(ROOT) and Path(origin)!=CODE:
            rel=Path(origin).relative_to(ROOT).as_posix();need(rel in f['dependencies']['files'],'loaded origin pinned');origins.append([name,rel])
    metadata=dict(schema='sealed-raw-model-v1',forwards=forwards,hook_parity=True,fixtures=tests,forward_seconds=timings,state_sha256=h.hexdigest(),final_state_sha256=final_state.hexdigest(),loaded_origin_indices=compact_origins(origins,f['dependencies']['files']),native_before_sha256=native_before['sha256'],torch_threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),cv_threads=cv2.getNumThreads(),architecture=architecture)
    logical=enc(dict(metadata,pairs=pairs));need(len(logical)<=int(2.5*MIB),'complete logical nonnative wire cap')
    core=dict(metadata=metadata,expected_pairs=20,pair_fragments=references,plain_sha256=plain_reference['sha256'],logical_model_sha256=sha(logical))
    model_part(req,'model-raw-v1',core)
    reconstructed,_=read_model_core();need(reconstructed==logical,'actual physical-fragment/logical-core roundtrip')
    native_after=native_images(f,req,'after')
    return dict(status='SEALED',pairs=20,anchors=2560,forwards=21,raw_sha256=sha(logical),report=actual(REPORT),native_before_sha256=native_before['sha256'],native_after_sha256=native_after['sha256'],native_validated=True,homography_values_read=False,prior_scored_manifest_parsed_for_array_hashes=True)



def summarized(rows,groups):
    folds={};signed=[]
    for group in groups:
        train=[r for r in rows if r['group']!=group];test=[r for r in rows if r['group']==group];table={}
        for k,arm in enumerate(ARMS):
            alpha=math.fsum(r['q'][k] for r in train)/(2*len(train));need(math.isfinite(alpha) and alpha>0,'undefined calibration scale')
            nll=[.5*(2*math.log(alpha)+r['logdet'][k]+r['q'][k]/alpha+2*math.log(2*math.pi)) for r in test]
            need(all(math.isfinite(x) for x in nll),'undefined nonfinite NLL')
            cov=[sum(r['q'][k]/alpha<=-2*math.log(1-q) for r in test)/len(test) for q in LEVELS]
            table[arm]=dict(alpha=alpha,mean_nll=math.fsum(nll)/len(test),coverage=cov,mean_absolute_coverage_error=math.fsum(abs(a-b) for a,b in zip(cov,LEVELS))/4,covariance_trace_range=[min(r['covariances'][k][0]+r['covariances'][k][2] for r in test),max(r['covariances'][k][0]+r['covariances'][k][2] for r in test)],covariance_determinant_range=[math.exp(min(r['logdet'][k] for r in test)),math.exp(max(r['logdet'][k] for r in test))])
        deltas={arm:table[arm]['mean_nll']-table['shape']['mean_nll'] for arm in ARMS[1:]};cal=table['constant']['mean_absolute_coverage_error']+1e-9-table['shape']['mean_absolute_coverage_error']
        signed.extend(x-1e-9 for x in deltas.values());signed.append(cal)
        folds[group]=dict(count=len(test),out_of_image_extrapolation_count=sum(r['outside'] for r in test),arms=table,nll_advantage=deltas,coverage_gate_margin=cal,pass_gate=all(x>1e-9 for x in deltas.values()) and cal>=0,mean_residual=[math.fsum(r['r'][k] for r in test)/640 for k in range(2)],gross_error_counts={str(t):sum(r['error']>t for r in test) for t in [1,5,10]})
    robust_fail=any(v < -1e-6 for v in signed);advance=all(v>1e-6 for v in signed);borderline=not robust_fail and not advance
    return dict(groups=folds,advance=advance,borderline=borderline,outcome='SCIENTIFIC_NEGATIVE' if robust_fail else 'QUALIFIED_PANEL_PROXY' if advance else 'INCONCLUSIVE',minimum_gate_distance=min(abs(v) for v in signed),signed_gate_margins=signed)


def scalar_audit(req):
    ownercheck(req);native_model_evidence(req['nonce']);sealed=latest('model-raw-v1');need(sealed['nonce']==req['nonce'],'sealed nonce');raw,model=read_model_core()
    idx=json.loads(readpin(INDEX));wire_panel(model,idx);rows=[];groups=sorted({p['source_group'] for p in idx['pairs']});undefined=[]
    for item,out in zip(idx['pairs'],model['pairs']):
        i=item['pair_index'];H=[list(map(float,line.split())) for line in readpin(item['homography'],links=2).decode().splitlines() if line.strip()];need(len(H)==3 and all(len(x)==3 for x in H),'H shape')
        A=[shapes(z) for z in out['logits0']];B=[shapes(z) for z in out['logits1']];perm=list(range(128));random.Random(20261024+i).shuffle(perm)
        for j in range(128):
            row=dict(pair=i,anchor=j,group=out['group'])
            try:
                u,v=out['out0'][j],out['out1'][j];target,J=project(H,u);r=[v[k]-target[k] for k in range(2)]
                source_inside=0<=u[0]<out['hw'][0][1] and 0<=u[1]<out['hw'][0][0];target_inside=0<=v[0]<out['hw'][1][1] and 0<=v[1]<out['hw'][1][0];reference_inside=0<=target[0]<out['hw'][1][1] and 0<=target[1]<out['hw'][1][0]
                a,b=A[j],B[j];ta=(a[0]+a[2])/2;tb=(b[0]+b[2])/2
                covs=[propagate(a,b,J),propagate([1,0,1],[1,0,1],J),propagate([ta,0,ta],[tb,0,tb],J),propagate(A[perm[j]],B[perm[j]],J)]
                qd=[quadratic(S,r) for S in covs]
                row.update(r=r,error=math.hypot(*r),outside=not(source_inside and target_inside and reference_inside),inside=[source_inside,target_inside,reference_inside],covariances=covs,q=[x[0] for x in qd],logdet=[x[1] for x in qd])
            except RuntimeError as e:
                row['undefined']=str(e);undefined.append(dict(pair=i,anchor=j,reason=str(e)))
            rows.append(row)
    need(len(rows)==2560 and len(groups)==4 and all(sum(r['group']==g for r in rows)==640 for g in groups),'all four full groups')
    result=dict(schema='independent-qualification-audit-v1',status='PASS',complete=True,nonce=req['nonce'],cases=2560,arms=list(ARMS),raw_sha256=sha(raw),no_new_pose_accuracy=True,rows=rows,undefined=undefined)
    if undefined:result.update(advance=False,borderline=False,outcome='NO_QUALIFICATION',groups={})
    else:
        try:result.update(summarized(rows,groups))
        except (RuntimeError,OverflowError,ValueError) as e:result.update(advance=False,borderline=False,outcome='NO_QUALIFICATION',groups={},undefined=[dict(scope='calibration',reason=str(e))])
    payload=enc(result);need(len(payload)<=2*MIB,'whole audit payload cap')
    append(dict(schema='audit-v1',nonce=req['nonce'],payload=packed(payload)),req['report_prefix'])
    return dict(status='AUDITED',complete=True,advance=result['advance'],outcome=result['outcome'],cases=2560,report=actual(REPORT),raw_sha256=sha(raw))


def root_confirmation(nonce):
    # Independent formulation: raw second moments, generic matrix products,
    # quotient-rule Jacobian, Cholesky solve and a second fold aggregation.
    native_model_evidence(nonce);raw,model=read_model_core();audit=json.loads(unpack(latest('audit-v1')['payload'],2*MIB));idx=json.loads(readpin(INDEX));wire_panel(model,idx)
    def moment(z):
        e=[math.exp(v-max(z)) for v in z];den=sum(e);w=[v/den for v in e];x=[j%3 for j in range(9)];y=[j//3 for j in range(9)];mx=sum(a*b for a,b in zip(w,x));my=sum(a*b for a,b in zip(w,y))
        return [[25*(sum(a*b*b for a,b in zip(w,x))-mx*mx),25*(sum(a*b*c for a,b,c in zip(w,x,y))-mx*my)],[25*(sum(a*b*c for a,b,c in zip(w,x,y))-mx*my),25*(sum(a*b*b for a,b in zip(w,y))-my*my)]]
    def matmul(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    def cov(A,B,J):
        V=matmul(matmul(J,A),[list(x) for x in zip(*J)]);return [[V[i][j]+B[i][j] for j in range(2)] for i in range(2)]
    rows=[];differences=[];unknown=[]
    for item,out in zip(idx['pairs'],model['pairs']):
        H=[list(map(float,line.split())) for line in readpin(item['homography'],links=2).decode().splitlines() if line.strip()];aa=[moment(v) for v in out['logits0']];bb=[moment(v) for v in out['logits1']];permutation=list(range(128));random.Random(20261024+item['pair_index']).shuffle(permutation)
        for j in range(128):
            record=dict(pair=item['pair_index'],anchor=j,group=item['source_group']);reference=audit['rows'][len(rows)];need(all(record[k]==reference[k] for k in record),'independent row identities')
            computation_error=None
            try:
                x,y=out['out0'][j];numerators=[row[0]*x+row[1]*y+row[2] for row in H];d=numerators[2];need(math.isfinite(d) and abs(d)>1e-12,'undefined projective denominator')
                mapped=[numerators[i]/d for i in (0,1)];J=[[(H[i][k]*d-numerators[i]*H[2][k])/(d*d) for k in (0,1)] for i in (0,1)];res=[out['out1'][j][k]-mapped[k] for k in (0,1)]
                a,b=aa[j],bb[j];at=(a[0][0]+a[1][1])/2;bt=(b[0][0]+b[1][1])/2
                matrices=[cov(a,b,J),cov([[1,0],[0,1]],[[1,0],[0,1]],J),cov([[at,0],[0,at]],[[bt,0],[0,bt]],J),cov(aa[permutation[j]],bb[permutation[j]],J)]
                q=[];ld=[]
                for k,S in enumerate(matrices):
                    need(all(math.isfinite(value) for line in S for value in line) and S[0][0]>0,'finite independent SPD entries')
                    l=math.sqrt(S[0][0]);v=S[1][0]/l;tail=S[1][1]-v*v;need(tail>0 and math.isfinite(tail),'undefined Cholesky');w=math.sqrt(tail);z0=res[0]/l;z1=(res[1]-v*z0)/w;q.append(z0*z0+z1*z1);ld.append(2*(math.log(l)+math.log(w)))
                    need(math.isfinite(q[-1]) and q[-1]>=0 and math.isfinite(ld[-1]),'finite independent Q/logdet')
                inside=[0<=x<out['hw'][0][1] and 0<=y<out['hw'][0][0],0<=out['out1'][j][0]<out['hw'][1][1] and 0<=out['out1'][j][1]<out['hw'][1][0],0<=mapped[0]<out['hw'][1][1] and 0<=mapped[1]<out['hw'][1][0]]
                record.update(q=q,logdet=ld,r=res,error=math.hypot(*res),inside=inside)
            except (RuntimeError,ValueError,OverflowError,ZeroDivisionError) as e:
                computation_error=str(e)
            # Only ROOT's own arithmetic can classify a row as undefined.
            # Reference-key access is deliberately outside that failure scope.
            need((computation_error is not None)==('undefined' in reference),'independent validity classification agrees')
            if computation_error is not None:
                record['undefined']=computation_error;unknown.append([record['pair'],j])
            else:
                need(inside==reference['inside'],'independent retained support flags')
                for k,S in enumerate(matrices):
                    differences.extend(abs(u-v) for u,v in zip([S[0][0],S[0][1],S[1][1]],reference['covariances'][k]));differences.extend([abs(q[k]-reference['q'][k]),abs(ld[k]-reference['logdet'][k])])
            rows.append(record)
    need(len(rows)==audit['cases']==2560,'independent denominator')
    folds={};signed=[]
    no_qualification=bool(unknown)
    if not unknown and audit['outcome']=='NO_QUALIFICATION':
        bad_scales=[]
        for group in sorted({r['group'] for r in rows}):
            training=[r for r in rows if r['group']!=group]
            held=[r for r in rows if r['group']==group]
            for k in range(4):
                scale=sum(r['q'][k] for r in training)/(2*len(training))
                if not math.isfinite(scale) or scale<=0:bad_scales.append([group,k,'scale'])
                else:
                    scores=[math.log(scale)+r['logdet'][k]/2+r['q'][k]/(2*scale)+math.log(2*math.pi) for r in held]
                    if not all(math.isfinite(value) for value in scores) or not math.isfinite(sum(scores)/640):bad_scales.append([group,k,'NLL'])
        need(bad_scales,'independent undefined calibration evidence');no_qualification=True;unknown.append(['calibration',bad_scales])
    if no_qualification:need(audit['outcome']=='NO_QUALIFICATION','undefined result joins')
    else:
        for group in sorted({r['group'] for r in rows}):
            train=[r for r in rows if r['group']!=group];test=[r for r in rows if r['group']==group];metrics={}
            for k,name in enumerate(ARMS):
                alpha=sum(r['q'][k] for r in train)/(2*len(train));need(math.isfinite(alpha) and alpha>0,'ROOT positive scale')
                scores=[math.log(alpha)+r['logdet'][k]/2+r['q'][k]/(2*alpha)+math.log(2*math.pi) for r in test]
                need(all(math.isfinite(value) for value in scores) and math.isfinite(sum(scores)/640),'finite ROOT NLL and aggregate')
                coverage=[sum(r['q'][k]/alpha<=-2*math.log1p(-p) for r in test)/640 for p in LEVELS];ce=sum(abs(a-b) for a,b in zip(coverage,LEVELS))/4
                metrics[name]=dict(alpha=alpha,mean_nll=sum(scores)/640,coverage=coverage,mean_absolute_coverage_error=ce)
                other=audit['groups'][group]['arms'][name];need(coverage==other['coverage'],'independent exact coverage counts');differences.extend([abs(alpha-other['alpha']),abs(metrics[name]['mean_nll']-other['mean_nll']),abs(ce-other['mean_absolute_coverage_error'])])
            signed.extend(metrics[name]['mean_nll']-metrics['shape']['mean_nll']-1e-9 for name in ARMS[1:]);signed.append(metrics['constant']['mean_absolute_coverage_error']+1e-9-metrics['shape']['mean_absolute_coverage_error']);folds[group]=metrics
        expected='SCIENTIFIC_NEGATIVE' if any(v< -1e-6 for v in signed) else 'QUALIFIED_PANEL_PROXY' if all(v>1e-6 for v in signed) else 'INCONCLUSIVE'
        need(audit['outcome']==expected and audit['advance']==(expected=='QUALIFIED_PANEL_PROXY'),'independent full decision')
    need(not differences or max(differences)<=1e-7,'independent numeric tolerance')
    result=dict(schema='root-confirmation-v1',nonce=nonce,complete=True,status='PASS',cases=2560,raw_sha256=sha(raw),max_absolute_numeric_difference=max(differences,default=0),folds=folds,outcome=audit['outcome'],advance=audit['advance'],undefined=unknown)
    append(result);return result


def run():
    started=time.monotonic();deadline=started+600;data_deadline=started+360
    signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,600)
    need(not any(r['schema']=='attempt-v1' for r in frames(closed=True)),'one fresh launch only')
    nonce=uuid.uuid4().hex;append(dict(schema='attempt-v1',nonce=nonce,started_monotonic=started,final_deadline=deadline,data_deadline=data_deadline,manager_pid=os.getpid()))
    proof_raw=sys.stdin.buffer.read(65537);need(len(proof_raw)<=65536,'bounded usage proof');proof=json.loads(proof_raw);need(set(ENV.items())<=set(os.environ.items()),'exact declared local environment')
    freeze_record=latest('freeze-v1');need(freeze_record['source']==actual(CODE) and freeze_record['command']==command('run'),'source and command frozen')
    need(roster()==freeze_record['roster'],'whole runtime/input readback and identity');res=census();health=host(proof)
    locks=[];reserved=False;known=True;observations=[];result_status='STOPPED_UNTESTED';engineering_complete=False;qualified=False;scientific_negative=False
    try:
        for n in ['research/experiment-owner.lock/stage.lock','research/managed-goal-20261010/publication.lock']:
            p=path(n);need(p.is_file() and p.stat().st_nlink==1,'existing lock inode');f=p.open('r+b');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        before,s,o=getshared();need([actual(ROOT/n) for n in SHARED]==freeze_record['shared'],'shared unchanged since freeze')
        b=s['budget'];need(KEY not in s and o['active_job'] is None and b['committed_experiment_wall_seconds']==0,'no duplicate/reservation')
        need(b['limit_experiment_wall_seconds']==137100 and b['failed_distinct_families']==8 and b['charged_experiment_wall_seconds']==122284.44731316697 and b['charged_experiment_wall_seconds']+600<=137100,'fixed budget affordability')
        originals=dict(schema='shared-originals-v1',nonce=nonce,paths=SHARED,raw=[packed(b) for b in before]);need(len(enc(originals))<=3*MIB,'complete original shared backup cap');append(originals)
        committed_prefix=actual(REPORT);source=actual(CODE);identity=subprocess.check_output(['/bin/ps','-p',str(os.getpid()),'-o','pid=,ppid=,pgid=,lstart=,command='],text=True,timeout=2).strip()
        job=dict(nonce=nonce,source=source,report_prefix=committed_prefix,manager_pid=os.getpid(),manager_identity=identity,started_monotonic=started,deadline=deadline,data_deadline=data_deadline,cap_seconds=600,status='ACTIVE')
        s[KEY]=dict(status='ACTIVE',inflight='qualification',job=job,offline_accepted=False,qualification_accepted=False)
        s['budget']['committed_experiment_wall_seconds']=600;o['active_job']=job;o['phase']='localization-uncertainty-qualification';o['updated_at']=utc()
        reserved=True;replace_shared(before,[enc(s),enc(o),before[2]+enc(dict(localization_uncertainty_reservation=job,utc=utc()))],'reserve',nonce)
        old=old_leaves();append(dict(schema='run-admission-v1',nonce=nonce,host=health,resources=res,original_started_monotonic=started,data_deadline=data_deadline,final_deadline=deadline))
        native_cap=None
        for mode,cap in [('probe',10),('count',20),('model',220),('audit',60)]:
            need(time.monotonic()+5<data_deadline,'remaining original data window');known=False
            req=dict(nonce=nonce,source=source,committed_prefix=committed_prefix,data_deadline=data_deadline,role=mode,report_prefix=None,native_count_cap=native_cap)
            append(dict(schema='launch-attempt-v1',nonce=nonce,role=mode,closure_known=False,remaining_data_seconds=data_deadline-time.monotonic()))
            class Launch(dict):
                def __setitem__(self,key,value):
                    super().__setitem__(key,value)
                    if key!='pid':return
                    table=old.ps_snapshot();row=table.get(value);need(row is not None and row['ppid']==os.getpid() and row['pgid']==value,'fresh direct actor identity')
                    full=subprocess.check_output(['/bin/ps','-ww','-p',str(value),'-o','command='],text=True,timeout=2).strip();args=self['argv'];runtime=freeze_record['roster']
                    need(full in {' '.join([first,*args[1:]]) for first in [str(PYTHON),PYREAL,runtime['main_app']['path']]},'exact actual actor argv/runtime')
                    child=dict(row,role=mode,argv=args,actual_command=full)
                    append(dict(schema='durable-child-v1',nonce=nonce,role=mode,child=child))
                    current,state,owner=getshared();need(owner['active_job']['nonce']==nonce and owner['active_job']['source']==source,'same manager ownership before child delivery')
                    state[KEY]['job']['child']=child;owner['active_job']['child']=child
                    replace_shared(current,[enc(state),enc(owner),current[2]],'child-'+mode,nonce)
                    req['report_prefix']=actual(REPORT)
            launch=Launch();launch['sample_limit']=256 if mode in ('probe','count') else 1200;seconds=min(cap,data_deadline-time.monotonic()-3);need(seconds>0,'positive bounded actor time')
            result=old.supervise(command('actor')+[mode],req,seconds,launch)
            need(len(enc(result))<=(MIB//2 if mode in ('probe','count') else MIB),'complete role-bounded actor observation');observations.append(dict(role=mode,launch=dict(launch),observation=result));known=result['owned_absence_confirmed']
            output_prefix=actual(REPORT);append(dict(schema='actor-observation-v1',nonce=nonce,role=mode,observation=result))
            need(known,'uncertain actor closure');need(result['accepted'],'actor failed')
            value=result['result'];need(value['role']==mode and value['nonce']==nonce,'returned actual role/nonce')
            if mode=='probe':need(value['status']=='HANDSHAKE' and value['fixtures']>=30 and value['report']==output_prefix,'real handshake/fixture result')
            elif mode=='count':
                need(value['report']==output_prefix,'actual count report prefix');d=validate_count_reply(value,{str(PYTHON),PYREAL,freeze_record['roster']['main_app']['path']})
                native_cap=native_capacity(d['count']);current,state,owner=getshared();need(owner['active_job']['nonce']==nonce,'count admission owns same reservation')
                state[KEY]['job']['native_count_cap']=native_cap;owner['active_job']['native_count_cap']=native_cap
                replace_shared(current,[enc(state),enc(owner),current[2]],'native-count-admission',nonce)
                append(dict(schema='native-capacity-admission-v2',nonce=nonce,observed_count=d['count'],old_512_predicate=d['old_512_predicate'],capacity=native_cap,rule='nextpow2(max512,n+256); rejectn0/>4096 andC>2048; neverclamp/reselect',forward_calls=0))
            elif mode=='model':
                raw,_=read_model_core()
                need(value['status']=='SEALED' and value['pairs']==20 and value['anchors']==2560 and value['forwards']==21 and value['raw_sha256']==sha(raw) and value['report']==output_prefix and value['homography_values_read'] is False and value['native_validated'] is True and value['native_before_sha256']==latest('native-before-v2')['payload']['sha256'] and value['native_after_sha256']==latest('native-after-v2')['payload']['sha256'],'model output saved-frame join');native_model_evidence(nonce)
            else:
                audit=json.loads(unpack(latest('audit-v1')['payload'],2*MIB))
                need(value['status']=='AUDITED' and value['complete'] is True and value['cases']==2560 and value['outcome']==audit['outcome'] and value['raw_sha256']==audit['raw_sha256'] and value['report']==output_prefix,'independent audit saved-frame join')
        need(time.monotonic()<data_deadline,'original data window before ROOT confirmation');confirmation=root_confirmation(nonce)
        need(roster()==freeze_record['roster'] and actual(CODE)==source,'terminal complete source/input/runtime/checkpoint readback');terminal_resources=census();need(terminal_resources['membership_sha256']==res['membership_sha256'],'terminal exact global membership');replay_transactions()
        append(dict(schema='terminal-validation-v1',nonce=nonce,resources=terminal_resources,source=source,roster_identity_sha256=freeze_record['roster']['runtime_identity_sha256'],all_attempted_groups_closed=True,elapsed_seconds=time.monotonic()-started))
        need(time.monotonic()<data_deadline,'complete data phase within original360')
        result_status=confirmation['outcome'];engineering_complete=True;qualified=confirmation['advance'];scientific_negative=result_status=='SCIENTIFIC_NEGATIVE'
    except BaseException as e:
        if not reserved:raise
        append(dict(schema='execution-error-v1',nonce=nonce,error=type(e).__name__+':'+str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        result_status='STOPPED_UNTESTED';engineering_complete=False;qualified=False;scientific_negative=False
        # Never replace current uncertainty with an older worker's closure.
        known=known and all(x['observation']['owned_absence_confirmed'] for x in observations)
    finally:
        if reserved:
            try:
                frames(closed=True);replay_transactions();before,s,o=getshared();need(o['active_job'] and o['active_job']['nonce']==nonce and s['budget']['committed_experiment_wall_seconds']==600,'exact live reservation')
                need(time.monotonic()<deadline,'original final settlement bound');need(known and not TEMP.exists(),'closure certain before settlement')
                for observation in observations:
                    pid=observation['observation']['pid'];table=old.ps_snapshot();need(pid not in table and not any(r['pgid']==pid for r in table.values()),'fresh all attempted actor-group absence')
                s['budget']['committed_experiment_wall_seconds']=0;s['budget']['charged_experiment_wall_seconds']+=600
                if scientific_negative:s['budget']['failed_distinct_families']+=1
                s[KEY].update(status=result_status,inflight=None,qualification_accepted=qualified,offline_accepted=engineering_complete,engineering_complete=engineering_complete,settled_cap_seconds=600,closure_known=True,elapsed_before_settlement=time.monotonic()-started,result_report=actual(REPORT))
                o['active_job']=None;o['phase']='localization-uncertainty-idle';o['updated_at']=utc()
                replace_shared(before,[enc(s),enc(o),before[2]+enc(dict(localization_uncertainty_settlement=nonce,status=result_status,charged=600,utc=utc()))],'settle',nonce)
                append(dict(schema='terminal-receipt-v1',nonce=nonce,status=result_status,qualification_accepted=qualified,engineering_complete=engineering_complete,elapsed_after_settlement=time.monotonic()-started,charged=600,committed=0,scientific_negative_counted=scientific_negative,budget=s['budget'],owner_idle=True))
                print(json.dumps(dict(status=result_status,qualified=qualified,engineering_complete=engineering_complete,elapsed_seconds=time.monotonic()-started,report=str(REPORT))))
            except BaseException as e:
                append(dict(schema='settlement-uncertain-v1',nonce=nonce,error=type(e).__name__+':'+str(e),elapsed_seconds=time.monotonic()-started));raise
        for f in reversed(locks):fcntl.flock(f.fileno(),fcntl.LOCK_UN);f.close()


if __name__=='__main__':
    need(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize,'isolated unoptimized interpreter')
    mode=sys.argv[1]
    if mode=='freeze':freeze()
    elif mode=='run':run()
    elif mode=='actor':
        role=sys.argv[2];fd=int(sys.argv[3]);deadline=float(sys.argv[4]);signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
        raw=sys.stdin.buffer.read(65537);need(len(raw)<=65536,'bounded request');req=json.loads(raw)
        if role=='probe':
            ownercheck(req);result=dict(status='HANDSHAKE',fixtures=len(fixtures()),report=actual(REPORT))
        else:result=model_actor(req) if role in ('model','count') else scalar_audit(req) if role=='audit' else None
        need(result is not None,'finite actor role');result.update(role=role,nonce=req['nonce']);body=enc(result);need(len(body)<=4096,'result pipe bound');os.write(fd,body);os.close(fd)
    else:raise RuntimeError('finite entrypoint')
