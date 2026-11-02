"""One bounded, human-requested evaluation of immutable sealed AR v8 evidence."""
import base64
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types
import zlib

ROOT = Path('/Users/ducng/Desktop/workspace/SLAMResaerch')
AR = 'research/ar-persistence-20261014/'
SOURCE = AR + 'evaluate-v9.py'
BUNDLE = AR + 'evaluation-lock-v9.json'
EVALUATION = AR + 'evaluation-v9.json'
JOB = AR + 'evaluation-job-v9.json'
BASE = dict(path=AR+'controller-v8.py', bytes=37614,
            sha256='3088d315e314645b573454afb96c63f62e397574fa7a2e78b53b0d7b1a8c631b')
PLAN_SHA = 'cc57f7c3d080160ce02797f1c82385d3571c7c6c0366e9240b16245d8bf94283'
REVIEWERS = {'ROOT', 'research_lessons', 'recent_status', 'verified_gains'}
CAP = 180
MIB = 1048576


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False)+'\n').encode('ascii')


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def base():
    path = ROOT/BASE['path']
    raw = path.read_bytes()
    assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
    assert len(raw) == BASE['bytes'] and hashlib.sha256(raw).hexdigest() == BASE['sha256']
    module = types.ModuleType('ar9_pinned_base')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def command(c, source):
    return dict(schema='ar-v9-command', stage='salvage',
        argv=[str(c.PYTHON), '-I', '-S', '-B', '-c', c.BOOT,
              str(ROOT/source['path']), str(source['bytes']), source['sha256'], 'stage', 'salvage'],
        environment=c.CLEAN_ENV, stdin='fresh ROOT ordinary-usage proof; no reused launch authority')


def package(c):
    value = c.document(BUNDLE, MIB)
    assert value['schema'] == 'ar-v9-final-lock'
    plan = value['prospective']
    assert digest(plan) == value['prospective_sha256'] == PLAN_SHA
    assert plan['budget']['cap_seconds'] == CAP
    assert plan['max_files'] == 128 and plan['max_bytes'] == 512*MIB
    assert value['source']['path'] == SOURCE and value['source']['bytes'] <= 65536
    c.read(value['source'], 65536)
    assert value['command'] == command(c, value['source'])
    for key in ('prospective_reviews', 'source_reviews'):
        rows = value[key]
        assert len(rows) == 4 and {row['reviewer'] for row in rows} == REVIEWERS
        for row in rows:
            assert row['decision'] == 'FULL_GO' and row['complete'] is True
            assert row['prospective_sha256'] == PLAN_SHA
            if key == 'source_reviews':
                assert row['source'] == value['source'] and row['command'] == value['command']
    for pin in plan['preserved']:
        c.read(pin, 16*MIB)
    for name in plan['retire_absent_permissions_only']:
        assert not c.physical(name, True).exists()
    c.read(plan['science']['lock']); c.read(plan['science']['passport'])
    assert not c.document(plan['science']['old_failure']['path'])['accepted']
    return value


def accounting(c, plan):
    d = c.disposition()
    d['namespace']['effective_caps'] = plan['effective_caps']
    return c.Accounting(d)


def actor(result_fd, deadline):
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    raw = sys.stdin.buffer.read(65537)
    assert len(raw) <= 65536
    request = json.loads(raw)
    c = base(); bundle = package(c); plan = bundle['prospective']
    admission = request['admission']
    assert digest(admission) == request['admission_sha256']
    owner = c.document(c.OWNER)
    assert owner['active_job']['admission'] == admission
    assert owner['active_job']['admission_sha256'] == digest(admission)
    assert admission['bundle'] == c.actual_pin(BUNDLE) and admission['source'] == bundle['source']
    assert admission['cap_seconds'] == CAP and admission['stage'] == 'EVALUATOR_ONLY_SALVAGE'
    def no_unapproved_action(event, args):
        if event in ('subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.system'):
            raise RuntimeError('evaluator cannot spawn processes')
        if event == 'import' and str(args[0]).split('.')[0] in ('pygcransac','torch'):
            raise RuntimeError('no model or fresh GC-fitting imports')
    sys.addaudithook(no_unapproved_action)
    started = time.perf_counter_ns()
    inherited = c.review()
    frozen = c.frozen_dependencies(inherited['dependencies'])
    old_freeze_pin = c.actual_pin(AR+'freeze-v8.json', 8*MIB)
    old_freeze = json.loads(c.read(old_freeze_pin, 8*MIB))
    assert frozen == old_freeze['dependencies']
    passport = c.document(plan['science']['passport']['path'])
    banks = [{k:b[k] for k in ('path','bytes','sha256')} for pair in passport['pairs'] for b in pair['banks']]
    for pin in banks: c.read(pin, 2*MIB)
    for pin in (passport['input_authority'], passport['data_manifest']): c.read(pin, 4*MIB)
    seal = c.document(plan['science']['seal']['path'])
    operands = [plan['science']['seal'], seal['prediction'], *seal['pose_pins']]
    assert len(operands) == 10
    for pin in operands: c.read(pin, 16*MIB)
    freeze = dict(schema='ar-v9-fresh-freeze-proof', old_inventory=old_freeze_pin,
        dependency_inventory_sha256=digest(frozen), dependency_file_count=len(frozen['files']),
        dependency_inventory_exactly_regenerated=True, sources=inherited['sources'],
        evaluator_source=bundle['source'], evaluator_bundle=admission['bundle'],
        bank_pins_sha256=digest(banks), bank_count=len(banks), sealed_operands=operands,
        python=sys.executable, environment=c.CLEAN_ENV, utc=c.utc(), actual_gt_read=False)
    sys.path[:] = [str(c.STDLIB), str(c.STDLIB/'lib-dynload'), str(c.SITE)]
    gate = c.FrozenGate(frozen['files']); sys.meta_path.insert(0, gate)
    producer = c.module_from_pin(inherited['sources'][0], 'ar9_pinned_producer')
    reference = c.module_from_pin(inherited['sources'][1], 'ar9_pinned_reference')
    context = dict(root=str(ROOT), version=8, seal_pin=plan['science']['seal'])
    result = run_science(context, producer, reference)
    assert result['audit']['status'] == 'PASS' and result['audit']['complete'] is True
    assert result['counts']['new_GC_fits'] == 0
    for pin in operands + banks: c.read(pin, 16*MIB)
    for pin in frozen['files'].values(): c.read(pin, 256*MIB)
    for pin in plan['preserved'] + [bundle['source'], admission['bundle'], passport['input_authority'], passport['data_manifest']]:
        c.read(pin, 16*MIB)
    c.read(dict(path='vendor/XFeat/assets/megadepth_1500.json',bytes=4024939,sha256='6aa01108aa26b1d5f5f1aad58a35514a173febd525f71cdab312ef72edbcca19'),4*MIB)
    assert c.verify_interpreter(inherited['dependencies']) == frozen['interpreter']
    c.verify_aliases(frozen['aliases'], frozen['files'])
    for module in tuple(sys.modules.values()):
        path = getattr(module,'__file__',None)
        if path and Path(path).is_relative_to(c.SITE): gate.verify(path)
    evaluation = dict(schema='ar-v9-salvaged-evaluation', source_version=8,
        evaluation_version=9, original_pilot=plan['science']['old_failure'],
        original_pilot_accepted=False, fresh_freeze=freeze, result=result,
        evaluator_elapsed_ns=time.perf_counter_ns()-started,
        original_producer_total_ns=None, original_producer_total_status='UNSAVED',
        resource_acceptance='PENDING_EXTERNAL_SUPERVISOR', full_pipeline_latency='UNMEASURED')
    body = encode(evaluation)
    assert len(body) <= MIB
    owner = c.document(c.OWNER)
    assert owner['active_job']['admission'] == admission and owner['active_job']['nonce'] == admission['nonce']
    assert owner['active_job']['admission_sha256'] == digest(admission)
    path = c.physical(EVALUATION, True)
    assert not path.exists()
    with path.open('xb') as stream: stream.write(body); stream.flush(); os.fsync(stream.fileno())
    assert path.read_bytes() == body
    artifact = c.pin(EVALUATION, body)
    envelope = dict(schema='ar-v9-actor-result', nonce=admission['nonce'], status='PASS', complete=True,
        artifact=artifact, outcome=result['summary']['outcome'], new_GC_fits=0)
    wire = encode(envelope)
    assert len(wire) <= 131072 and time.monotonic() < deadline
    view = memoryview(wire)
    while view: view = view[os.write(result_fd, view):]
    os.close(result_fd)


def stage(proof):
    c = base(); bundle = package(c); plan = bundle['prospective']; a = accounting(c, plan)
    state, owner = c.state_owner(); branch = state['ar_persistence']
    assert set(branch['jobs']) == {f'{i:02d}' for i in range(11)}
    assert branch['jobs']['10']['accepted'] is False
    old_job10 = copy.deepcopy(branch['jobs']['10'])
    assert branch['charged_seconds'] == 1020 and branch['job_allowance_seconds'] == 1200
    assert state['budget']['charged_experiment_wall_seconds'] == 24870.44731316697
    assert state['budget']['failed_distinct_families'] < state['budget']['limit_failed_distinct_families']
    for pin in plan['shared_start']: c.read(pin)
    additions = {EVALUATION:MIB, JOB:MIB}
    forecast = a.forecast(additions)
    host = c.resources(proof)
    nonce = c.uuid.uuid4().hex
    admission = dict(schema='ar-v9-admission', stage='EVALUATOR_ONLY_SALVAGE', version=9,
        slot='11', nonce=nonce, cap_seconds=CAP, bundle=c.actual_pin(BUNDLE), source=bundle['source'],
        command=bundle['command'], resources=host, forecast=forecast,
        prior_budget=copy.deepcopy(state['budget']), original_job10=old_job10, utc=c.utc())
    owner['active_job'] = dict(run_id='ar-persistence-20261014', version=9,
        stage='EVALUATOR_ONLY_SALVAGE', nonce=nonce, cap_seconds=CAP,
        admission=admission, admission_sha256=digest(admission))
    owner['phase'] = 'ar-v9-evaluator-active'
    state['budget']['committed_experiment_wall_seconds'] += CAP
    branch['reserved_seconds'] = CAP; branch['status'] = 'RUNNING_EVALUATOR_ONLY'
    state['updated_at'] = owner['updated_at'] = c.utc()
    a.forecast(additions, {c.STATE:encode(state), c.OWNER:encode(owner)})
    a.shared(state, owner)
    observed = None; launch = {}; accepted = False; error = None; artifact = None
    try:
        c.resources(proof)
        src = bundle['source']
        argv = [str(c.PYTHON),'-I','-S','-B','-c',c.BOOT,
            str(ROOT/src['path']),str(src['bytes']),src['sha256'],'actor','salvage']
        observed = c.supervise(argv, dict(admission=admission,admission_sha256=digest(admission)), CAP, launch)
        value = observed['result']
        assert observed['accepted'], observed['errors']
        assert value['schema']=='ar-v9-actor-result' and value['nonce']==nonce
        assert value['status']=='PASS' and value['complete'] is True and value['new_GC_fits']==0
        artifact = value['artifact']; assert artifact['path'] == EVALUATION
        report = json.loads(c.read(artifact, MIB))
        assert report['result']['audit']['status']=='PASS' and report['result']['audit']['complete'] is True
        assert value['outcome'] == report['result']['summary']['outcome']
        for pin in plan['preserved'] + [bundle['source'], admission['bundle']]: c.read(pin,16*MIB)
        accepted = True
    except BaseException as exc:
        error = type(exc).__name__+':'+str(exc)[:3000]
    launched = 'pid' in launch
    assert not launched or observed is not None and observed['owned_absence_confirmed'], 'unknown closure: retain owner/reservation'
    if observed is not None:
        samples=observed.pop('sampled_resources')
        observed['sampled_resources_archive']=dict(encoding='base64(zlib(canonical JSON))',count=len(samples),
            sha256=digest(samples),content=base64.b64encode(zlib.compress(encode(samples),9)).decode())
    record=dict(schema='ar-v9-job-result',version=9,stage='EVALUATOR_ONLY_SALVAGE',utc=c.utc(),
        accepted=accepted,launched=launched,error=error,admission=admission,
        admission_sha256=digest(admission),actor=observed,artifact=artifact,
        full_reserved_cap_charged=CAP if launched else 0,original_job10_accepted=False,
        scientific_status='INDEPENDENTLY_SCORED_PENDING_SETTLEMENT' if accepted else 'UNTESTED')
    raw=encode(record); assert len(raw)<=MIB
    rp=a.create(JOB,raw)
    state=c.document(c.STATE);owner=c.document(c.OWNER);branch=state['ar_persistence']
    assert owner['active_job']['nonce']==nonce and owner['active_job']['admission_sha256']==digest(admission)
    assert branch['jobs']['10']==old_job10
    fullcap=CAP if launched else 0
    state['budget']['committed_experiment_wall_seconds']-=CAP
    state['budget']['charged_experiment_wall_seconds']+=fullcap
    branch['reserved_seconds']=0;branch['charged_seconds']+=fullcap
    branch['jobs']['11']=dict(version=9,kind='EVALUATOR_ONLY_SALVAGE',stage='evaluation',
        accepted=accepted,launched=launched,result=rp,admission_sha256=digest(admission))
    branch['v9_evaluation_lock']=admission['bundle']
    branch['status']='ENGINEERING_FAILURE'
    if accepted:
        outcome=report['result']['summary']['outcome']
        assert outcome in ('PILOT_PASS_PENDING_AUDIT','SCIENTIFIC_NEGATIVE_PENDING_AUDIT','NO_EFFECT_PENDING_AUDIT')
        assert not branch.get('family_counted',False)
        negative=outcome!='PILOT_PASS_PENDING_AUDIT'
        if negative:
            state['budget']['failed_distinct_families']+=1;branch['family_counted']=True
        branch['scientific_result']='SCIENTIFIC_NEGATIVE' if negative else 'PILOT_PASS'
        branch['status']='SALVAGED_EVIDENCE_AUDITED';branch['v9_evaluation']=artifact
        branch['v9_family_increment']=int(negative)
    assert branch['charged_seconds']<=1200 and state['budget']['charged_experiment_wall_seconds']<=86400
    assert state['budget']['charged_experiment_wall_seconds']==23850.44731316697+branch['charged_seconds']
    owner['active_job']=None;owner['phase']='ar-v9-evaluator-closed'
    state['updated_at']=owner['updated_at']=c.utc()
    note=f'\n\nAR v9 evaluator-only {c.utc()}: accepted={accepted}, launched={launched}, charged={fullcap}s; AR={branch["charged_seconds"]}/1200/global={state["budget"]["charged_experiment_wall_seconds"]}/families={state["budget"]["failed_distinct_families"]}; original v8 job10 remains rejected. Receipt {JOB}. No further job admitted.\n'.encode()
    a.shared(state,owner,c.read(c.actual_pin(c.BOOK),MIB)+note)
    print(encode(dict(accepted=accepted,launched=launched,error=error,result=rp,
        artifact=artifact,scientific_result=branch['scientific_result'],AR_charged=branch['charged_seconds'])).decode(),flush=True)
    if not accepted: raise SystemExit(1)


# SOURCE ONLY science fragment; importing defines functions and string constants only.
# The caller binds immutable v8 modules and controls all I/O/resource admission.
AR_V9_PRODUCER_GT_SOURCE = r"""def ground_truth(root, pairs):
    import numpy as np
    raw = (Path(root) / GT).read_bytes()
    require(sha(raw) == GT_SHA, 'GT authority hash')
    data = json.loads(raw)
    require(type(data) is list and len(data) == 1500, 'GT master schema')
    output = []
    for pair in pairs:
        indices = [i for i, record in enumerate(data) if record['pair_names'] == pair['pair_names']]
        require(len(indices) == 1, 'unique directed GT join')
        i = indices[0]
        require(type(data[i]['pair_id']) is int and
                type(pair['legacy_metadata_pair_id']) is int and
                data[i]['pair_id'] == pair['legacy_metadata_pair_id'], 'GT pair ID join')
        require(data[i]['scene_id'] == pair['scene'] + '_0', 'GT scene join')
        for key in ('K0', 'K1'):
            require(np.array_equal(np.asarray(data[i][key]), np.asarray(pair[key])), 'GT calibration join')
        T = np.asarray(data[i]['T_0to1'], dtype=np.float64)
        require(T.shape == (4, 4) and np.isfinite(T).all() and
                np.array_equal(T[3], [0., 0., 0., 1.]), 'directed transform schema')
        output.append(T)
    return output
"""

AR_V9_REFERENCE_GT_SOURCE = r"""def _directed_truths(context, pairs):
    import numpy as np
    pin = dict(path='vendor/XFeat/assets/megadepth_1500.json', bytes=4024939,
               sha256='6aa01108aa26b1d5f5f1aad58a35514a173febd525f71cdab312ef72edbcca19')
    raw = _read(context, pin, 4194304)
    data = json.loads(raw.decode('utf8'))
    assert type(data) is list and len(data) == 1500
    truths = []
    for pair in pairs:
        locations = [i for i, item in enumerate(data) if item['pair_names'] == pair['pair_names']]
        assert len(locations) == 1
        record = data[locations[0]]
        assert type(record['pair_id']) is int and type(pair['legacy_metadata_pair_id']) is int
        assert record['pair_id'] == pair['legacy_metadata_pair_id']
        assert record['scene_id'] == pair['scene'] + '_0'
        assert np.array_equal(np.asarray(record['K0']), np.asarray(pair['K0']))
        assert np.array_equal(np.asarray(record['K1']), np.asarray(pair['K1']))
        transform = np.asarray(record['T_0to1'], dtype=np.float64)
        assert transform.shape == (4, 4) and np.isfinite(transform).all()
        assert np.array_equal(transform[3], np.array([0., 0., 0., 1.]))
        assert _proper(transform[:3, :3])
        truths.append(transform)
    return truths
"""

def _ar_v9_install_gt(producer, reference):
    """Replace exactly the two GT functions in already hash-bound v8 modules."""
    import hashlib
    import types
    before = [{k: v for k, v in module.__dict__.items()
               if isinstance(v, types.FunctionType)} for module in (producer, reference)]
    exec(compile(AR_V9_PRODUCER_GT_SOURCE, '<ar-v9-producer-gt>', 'exec'), producer.__dict__)
    exec(compile(AR_V9_REFERENCE_GT_SOURCE, '<ar-v9-reference-gt>', 'exec'), reference.__dict__)
    for module, old, changed in zip((producer, reference), before,
                                    ('ground_truth', '_directed_truths')):
        now = {k: v for k, v in module.__dict__.items() if isinstance(v, types.FunctionType)}
        assert set(now) == set(old)
        assert all(now[k] is value for k, value in old.items() if k != changed)
        assert now[changed] is not old[changed]
    return dict(producer_gt_sha256=hashlib.sha256(AR_V9_PRODUCER_GT_SOURCE.encode()).hexdigest(),
        reference_gt_sha256=hashlib.sha256(AR_V9_REFERENCE_GT_SOURCE.encode()).hexdigest(),
        all_other_function_objects_unchanged=True)


def _ar_v9_gt_fixtures(producer, reference):
    """Exercise both actual corrected GT functions on in-memory JSON only."""
    import copy
    import hashlib
    import json
    camera0 = [[100., 0., 10.], [0., 110., 11.], [0., 0., 1.]]
    camera1 = [[120., 0., 12.], [0., 130., 13.], [0., 0., 1.]]
    transform = [[1., 0., 0., 1.], [0., 1., 0., 0.],
                 [0., 0., 1., 0.], [0., 0., 0., 1.]]
    inverse = [[1., 0., 0., -1.], [0., 1., 0., 0.],
               [0., 0., 1., 0.], [0., 0., 0., 1.]]
    pair = dict(scene='0015', legacy_metadata_pair_id=7,
        pair_names=['fixture/source.jpg', 'fixture/target.jpg'], K0=camera0, K1=camera1)
    master = [dict(pair_names=[f'fixture/a{i}.jpg', f'fixture/b{i}.jpg'],
        pair_id=i, scene_id='other_0', K0=copy.deepcopy(camera0),
        K1=copy.deepcopy(camera1), T_0to1=copy.deepcopy(transform),
        T_1to0=copy.deepcopy(inverse)) for i in range(1500)]
    target = dict(pair_names=pair['pair_names'][:], pair_id=7, scene_id='0015_0',
        K0=copy.deepcopy(camera0), K1=copy.deepcopy(camera1),
        T_0to1=copy.deepcopy(transform), T_1to0=copy.deepcopy(inverse))
    master[103] = copy.deepcopy(target)
    master[7] = dict(copy.deepcopy(target), pair_names=['fixture/decoy0', 'fixture/decoy1'])
    orders = (list(range(1500)), list(reversed(range(1500))),
              list(range(347, 1500)) + list(range(347)))
    cases = [(f'valid_fixed_order_{i}', [master[j] for j in order], True)
             for i, order in enumerate(orders)]
    for defect in ('duplicate_names', 'reversed_only', 'wrong_id', 'wrong_scene',
                   'wrong_K0', 'wrong_K1', 'float_id', 'bool_id'):
        data = copy.deepcopy(master)
        if defect == 'duplicate_names':
            data[104] = copy.deepcopy(target)
        elif defect == 'reversed_only':
            data[103]['pair_names'].reverse()
        elif defect == 'wrong_id':
            data[103]['pair_id'] = 8
        elif defect == 'wrong_scene':
            data[103]['scene_id'] = '0022_0'
        elif defect in ('wrong_K0', 'wrong_K1'):
            data[103][defect[6:]][0][0] += 1.
        elif defect == 'float_id':
            data[103]['pair_id'] = 7.0
        else:
            data[103]['pair_id'] = True
        cases.append((defect, data, False))
    old_path, old_sha, old_read = producer.Path, producer.GT_SHA, reference._read
    raw = b''
    fixture_root = '__ar_v9_in_memory_fixture__'

    class FixturePath:
        def __init__(self, name):
            assert name == fixture_root
        def __truediv__(self, name):
            assert name == producer.GT
            return self
        def read_bytes(self):
            return raw

    def fixture_read(context, pin, cap):
        assert context == {'root': fixture_root}
        assert pin == dict(path=producer.GT, bytes=4024939, sha256=old_sha)
        assert cap == 4194304
        return raw

    checks = []
    try:
        producer.Path, reference._read = FixturePath, fixture_read
        for name, data, valid in cases:
            raw = (json.dumps(data, sort_keys=True, separators=(',', ':'),
                              allow_nan=False) + '\n').encode()
            producer.GT_SHA = hashlib.sha256(raw).hexdigest()
            for loader, call in (
                ('producer', lambda: producer.ground_truth(fixture_root, [pair])),
                ('reference', lambda: reference._directed_truths({'root': fixture_root}, [pair]))):
                if valid:
                    values = call()
                    assert len(values) == 1 and values[0].tolist() == transform
                else:
                    try:
                        call()
                    except (AssertionError, ValueError):
                        pass
                    else:
                        raise AssertionError('Malformed GT fixture accepted: ' + loader + ':' + name)
                checks.append(dict(id=loader + ':' + name, status='PASS', expected_accept=valid))
    finally:
        producer.Path, producer.GT_SHA, reference._read = old_path, old_sha, old_read
    assert producer.Path is old_path and producer.GT_SHA == old_sha
    assert reference._read is old_read
    return dict(schema='ar-v9-gt-fixtures', status='PASS', complete=True,
        fabricated_master_rows=1500, cases_per_loader=len(cases), checks=checks,
        real_GT_reads=0, fixture_files_written=0, all_stubs_restored=True)


def run_science(context, producer, reference):
    """Validate and score immutable v8 evidence; writes no files and performs no GC fit."""
    import time
    begin = time.perf_counter_ns()
    assert type(context['version']) is int and context['version'] == 8
    bindings = _ar_v9_install_gt(producer, reference)
    fixture_begin = time.perf_counter_ns()
    fixtures = _ar_v9_gt_fixtures(producer, reference)
    fixture_ns = time.perf_counter_ns() - fixture_begin
    validation_begin = time.perf_counter_ns()
    version = context['version']
    passport = reference._document(context, reference.PASSPORT_PIN)
    assert len(passport['pairs']) == 8
    seal_name = reference.RESULT_ROOT + 'prediction-seal-v8.json'
    seal_pin, seal_raw = reference._existing(context, seal_name)
    assert seal_pin == context['seal_pin']
    seal = reference.json.loads(seal_raw.decode('ascii'))
    assert reference._json(seal) == seal_raw
    assert set(seal) == {'schema', 'prediction', 'pose_pins', 'gt_read', 'status'}
    assert seal['schema'] == 'ar-prediction-seal-v1' and seal['gt_read'] is False
    assert seal['prediction']['path'] == reference.RESULT_ROOT + 'predictions-v8.npz'
    prediction, prediction_arrays = reference._record(context, seal['prediction'])
    assert set(prediction_arrays) == {'record_json'}
    assert prediction['schema'] == 'ar-predictions-v1' and type(prediction['version']) is int
    assert prediction['version'] == version and type(prediction['native_calls']) is int
    assert prediction['labels_read'] is False and prediction['native_calls'] == 0
    assert prediction['input_pins'] == [bank for pair in passport['pairs'] for bank in pair['banks']]
    records = prediction['pairs']
    assert len(records) == 8
    all_banks, input_replay = [], []
    for p, (pair, record) in enumerate(zip(passport['pairs'], records)):
        assert type(pair['pair_index']) is type(record['pair_index']) is int
        assert pair['pair_index'] == record['pair_index'] == p
        assert record['scene'] == pair['scene']
        banks = reference.load_pair(context, pair)
        features, camera, _, fast = banks
        expected = reference.filter_pair(features['descriptors0'], features['descriptors1'],
            fast['source_indices'], fast['target_indices'], fast['raw_p0'], camera['hw0'])
        reference._compare_filter(record['primary'], expected)
        reference._check_assignment_witness(record['primary'], features['descriptors0'],
            features['descriptors1'], fast['source_indices'], fast['target_indices'])
        assert len(record['solve_counts']) == 6
        assert all(type(v) is int and 1 <= v <= 33 for v in record['solve_counts'])
        assert record['solve_counts'] == [record['primary']['solves']] * 6
        assert type(record['input_load_and_integrity_ns']) is int
        assert record['input_load_and_integrity_ns'] >= 0
        input_replay.append(dict(pair_index=p, q=expected['q'], rows=len(expected['rows']),
            cols=len(expected['cols']), solves=expected['solves'], masks_match=True,
            objective_match=True, reference_uses_producer_math=False,
            q_bins=record['primary']['q_bins'], anchors=record['primary']['anchors'],
            deltas=record['primary']['deltas'], optimum=record['primary']['optimum'],
            original_input_load_and_integrity_ns=record['input_load_and_integrity_ns']))
        all_banks.append(banks)
    producer_solves = sum(sum(record['solve_counts']) for record in records)
    validation_solves = sum(row['solves'] for row in input_replay)
    assert producer_solves <= 1584 and validation_solves <= 264
    ineffective, differences = reference.no_effect(records)
    timing = reference._timings(records)
    # This finite salvage is for the exact already-pinned all-native seal only.
    assert not ineffective, 'Unexpected no-effect branch for sealed native evidence'
    assert seal['status'] == 'ALL_NATIVE_OUTPUTS_SEALED' and len(seal['pose_pins']) == 8
    poses = []
    for p, pose_pin in enumerate(seal['pose_pins']):
        assert pose_pin['path'] == reference.RESULT_ROOT + f'poses-v8-{p:04d}.npz'
        pose, arrays = reference._record(context, pose_pin)
        assert pose['schema'] == 'ar-native-pose-v1' and type(pose['pair_index']) is int
        assert pose['pair_index'] == p and pose['complete'] is True and 'failure' not in pose
        assert pose['prediction'] == seal['prediction']
        assert type(pose['fits']) is list and len(pose['fits']) <= 30
        assert all(type(fit['recoveries']) is list for fit in pose['fits'])
        poses.append((pose, arrays))
    planned_recoveries = sum(len(fit['recoveries']) for pose, _ in poses for fit in pose['fits'])
    assert planned_recoveries <= 240, 'Native recovery replay ceiling'
    native_replay, attempts = [], 0
    for p, ((pose, arrays), pose_pin) in enumerate(zip(poses, seal['pose_pins'])):
        groups = [group for draw in range(5) for group in reference._expected_groups(
            p, draw, all_banks[p], records[p]['primary']['masks'])]
        assert len(groups) == len(pose['fits'])
        used = set()
        for fit, group in zip(pose['fits'], groups):
            for name in ('id', 'draw', 'first_arm_position', 'arms', 'coords_hash'):
                assert fit[name] == group[name]
            result = reference._check_fit(fit, arrays, used, group['p0'], group['p1'], all_banks[p][1])
            attempts += result['attempted']
            native_replay.append(dict(pair_index=p, fit_id=fit['id'],
                producer_pose_bank=pose_pin, selected_index=result['selected'],
                original_fit_timing_ns=fit['timing_ns'], original_GC_timing_ns=fit.get('gc_ns'),
                original_recovery_timings_ns=[r['native_ns'] for r in fit['recoveries']],
                replays=result['replays'], complete=True))
        assert used == set(arrays) - {'record_json'}
    assert attempts <= 240
    recovered_replays = sum(len(row['replays']) for row in native_replay)
    assert recovered_replays == planned_recoveries
    validation_ns = time.perf_counter_ns() - validation_begin
    # First actual GT access in THIS evaluator, after every saved native output validates.
    producer_score_begin = time.perf_counter_ns()
    producer_truths = producer.ground_truth(context['root'], passport['pairs'])
    produced = producer.score(poses, producer_truths, records, timing)
    producer_scoring_ns = time.perf_counter_ns() - producer_score_begin
    reference_score_begin = time.perf_counter_ns()
    reference_truths = reference._directed_truths(context, passport['pairs'])
    assert len(producer_truths) == len(reference_truths) == 8
    assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
               for a, b in zip(producer_truths, reference_truths))
    scored = reference._score(poses, reference_truths, records, timing)
    reference._nested_match(produced, scored)
    reference_scoring_ns = time.perf_counter_ns() - reference_score_begin
    readback_begin = time.perf_counter_ns()
    reference._read(context, seal_pin, 1048576)
    reference._read(context, seal['prediction'], 16777216)
    for operand in seal['pose_pins']:
        reference._read(context, operand, 16777216)
    reference._read(context, reference.PASSPORT_PIN, 1048576)
    reference._read(context, dict(path=producer.GT, bytes=4024939,
        sha256=producer.GT_SHA), 4194304)
    readback_ns = time.perf_counter_ns() - readback_begin
    summary = dict(schema='ar-v9-sealed-evidence-summary', source_version=version,
        evaluation_version=9, original_pilot_accepted=False,
        evaluation_mode='EVALUATOR_ONLY_SALVAGE',
        external_acceptance='PENDING_SUPERVISOR_ACCEPTANCE',
        prediction_pin=seal['prediction'], seal_pin=seal_pin, pose_pins=seal['pose_pins'],
        mask_differences=differences, filter_timing=timing, gt_read=True,
        original_GC_fits=attempts, new_GC_fits=0, recovery_replays=recovered_replays,
        full_pipeline_latency='UNMEASURED', physical_correspondence_truth='UNKNOWN',
        scientific_family_increment='ONLY_AFTER_ACCEPTED_INDEPENDENT_EVALUATION',
        original_unsaved_timings=dict(status='UNSAVED', producer_elapsed_ns=None,
            numerical_import_ns=None, passport_load_ns=None,
            prelabel_readback_ns=None, final_summary_io_ns=None),
        original_scoring_status='NOT_EXECUTED_BEFORE_GT_JOIN_FAILURE', **produced)
    audit = dict(schema='ar-v9-sealed-independent-replay', status='PASS', complete=True,
        source_version=version, evaluation_version=9, prediction_seal=seal_pin,
        prediction=seal['prediction'], producer_pose_banks=seal['pose_pins'],
        mask_pairs=8, input_replay=input_replay, original_native_fits=attempts,
        new_GC_fits=0, recovered_replays=recovered_replays, native_replay=native_replay,
        summary_verified=True, both_original_scorers_agree=True, no_effect=False,
        outcome=produced['outcome'], pair_match_counts_and_overlap=reference._mask_reports(records),
        tolerance=dict(objective=1e-10, recovery=1e-9, score=1e-8, rtol=0),
        raw_evidence_scope='whole immutable v8 producer returns; independent schemas/hash evidence',
        unused_fifth_policy='preserved losslessly; not selected/scored; nonfinite values allowed',
        physical_correspondence_truth='UNKNOWN', full_pipeline_latency='UNMEASURED',
        end_rehash_scope='seal/prediction/eight poses/passport/GT; wrapper also rehashes banks/dependencies')
    counts = dict(pairs=8, arms=6, draws=5,
        logical_arm_draw_slots=sum(len(fit['arms']) for pose, _ in poses for fit in pose['fits']),
        original_filter_evaluations=sum(len(record['solve_counts']) for record in records),
        original_timed_filter_samples=sum(len(record['timed_ns']) for record in records),
        original_assignment_solves=producer_solves, validation_assignment_solves=validation_solves,
        original_GC_fits=attempts, original_saved_recoveries=planned_recoveries,
        new_GC_fits=0, recovery_replays=recovered_replays, new_filter_timing_runs=0)
    assert counts['logical_arm_draw_slots'] == 240
    assert counts['original_filter_evaluations'] == 48
    assert counts['original_timed_filter_samples'] == 40
    return dict(fixtures=fixtures, summary=summary, audit=audit, counts=counts,
        evaluator=dict(schema='ar-v9-science-provenance', status='PASS', complete=True,
            corrected_GT_bindings=bindings, GT_before_real_native_validation=False,
            inherited_GT_access='v8 actor already parsed GT after its output seal; no scores retained',
            original_summary_written=False, old_artifacts_modified=False,
            timings_ns=dict(fabricated_fixtures=fixture_ns,
                artifact_math_native_validation=validation_ns,
                producer_GT_and_scoring_replay=producer_scoring_ns,
                independent_GT_and_scoring=reference_scoring_ns,
                final_sealed_operand_readback=readback_ns,
                total_before_return=time.perf_counter_ns() - begin),
            timings_scope='new evaluator wall spans; not new filter/native fitting performance samples'))

if __name__ == '__main__':
    assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize
    mode, kind = sys.argv[1:3]
    assert kind == 'salvage'
    if mode == 'actor':
        actor(int(sys.argv[3]), float(sys.argv[4]))
    else:
        assert mode == 'stage'
        raw = sys.stdin.buffer.read(65537)
        assert len(raw) <= 65536
        proof = json.loads(raw)
        locks = []
        try:
            for name in ('research/experiment-owner.lock/stage.lock', 'research/managed-goal-20261010/publication.lock'):
                stream = (ROOT/name).open('r+b')
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
                locks.append(stream)
            stage(proof)
        finally:
            for stream in locks: stream.close()
