"""RM051 scientific stage adapter over executed Small/native/scoring seams.

Only execute(c, spec) is an entry point. The owning parent authenticates the
source/config checkpoint, owns the stable lock/resources, saves receipts and
builds whole-panel phase seals. No legacy controller or campaign main is used.
"""
from pathlib import Path
import ast
import csv
import ctypes
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import stat
import sys
import time
import types

PROJECT = Path('/Users/ducng/Desktop/workspace/SLAMResaerch')
ARMS = ['star_small', 'anchor_consensus', 'ungated_medoid', 'recrop_median',
        'fixed_recrop', 'branch_shuffled', 'star_native', 'stock_small']
BRANCH_ORDER = ['B', '+x0', '-x0', '+y0', '-y0', '+x1', '-x1', '+y1', '-y1']
SHIFTS = [(None, None, 0), (0, 0, 1), (0, 0, -1), (0, 1, 1),
          (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 1), (1, 1, -1)]
MIB = 1048576


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def pin(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
             'regular single-link stage input')
        h = hashlib.sha256()
        while True:
            block = os.read(fd, 262144)
            if not block:
                break
            h.update(block)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
                           s.st_ctime_ns, s.st_nlink, s.st_mode)
        need(stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()),
             'stage input drift')
        return dict(path=str(path.absolute()), bytes=before.st_size,
                    sha256=h.hexdigest())
    finally:
        os.close(fd)


def read(record, cap=64 * MIB):
    need(type(record) is dict and set(record) == {'path', 'bytes', 'sha256'}
         and type(record['bytes']) is int and 0 <= record['bytes'] <= cap,
         'flat bounded input pin')
    need(pin(record['path']) == record, 'exact stage input pin')
    body = Path(record['path']).read_bytes()
    need(len(body) == record['bytes']
         and hashlib.sha256(body).hexdigest() == record['sha256'],
         'same authenticated input buffer')
    return body


def load(record, name):
    body = read(record)
    module = types.ModuleType(name)
    module.__file__ = record['path']
    sys.modules[name] = module
    exec(compile(body, record['path'], 'exec', dont_inherit=True), module.__dict__)
    need(pin(record['path']) == record, 'loaded stage source drift')
    return module


def create(c, path, body, cap=64 * MIB):
    root, path = Path(c['root']), Path(path)
    need(root.is_absolute() and path.is_absolute() and '..' not in path.parts
         and root in path.parents and len(body) <= cap,
         'new owned stage output')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(not any(p.is_symlink() for p in (path, *path.parents)),
         'physical owned output path')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        need(stream.write(body) == len(body), 'complete stage artifact write')
        stream.flush()
        os.fsync(stream.fileno())
    result = pin(path)
    need(read(result, cap) == body, 'full stage artifact readback')
    return result


def save_npz(c, np, path, arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return create(c, path, buffer.getvalue())


def reader(c):
    name = '_rm051_existing_saved_reader'
    module = sys.modules.get(name)
    if module is None:
        module = load(c['sources']['reader'], name)
    need(module.__file__ == c['sources']['reader']['path'], 'exact reader source')
    return module


def verify_payload(c, record):
    if 'packed' in record:
        rr = reader(c).Reader([record['packed']['container']],
                              Path(c['root']) / 'unused-reader-output')
        rr.verify(record)
    else:
        read(record)


def arrays(c, np, record):
    if 'packed' in record:
        rr = reader(c).Reader([record['packed']['container']],
                              Path(c['root']) / 'unused-reader-output')
        return {k: np.array(a.values(), dtype=a.dtype).reshape(a.shape)
                for k, a in rr.arrays(record).items()}
    with np.load(io.BytesIO(read(record)), allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def plan_document(c, spec):
    plan = json.loads(read(spec['plan'], 32 * MIB))
    need(plan['methods'] == ARMS and plan['baseline'] == 'star_small'
         and plan['policies'] == ['legacy', 'conditional_depth']
         and plan['primary_policy'] == 'conditional_depth'
         and plan['normalization'] == 'corrected', 'frozen RM051 roles/policy')
    need(type(plan['pairs']) is list and len(plan['pairs']) == len(plan['records'])
         and all(type(v) is int for v in plan['pairs'])
         and len(set(plan['pairs'])) == len(plan['pairs'])
         and plan['pairs'] == [r['pair_index'] for r in plan['records']],
         'complete ordered original directed pair IDs')
    output = Path(plan['output'])
    need(Path(c['root']) in output.parents and output.is_absolute(),
         'new owned panel namespace')
    return plan


def indices_for(spec, plan):
    start, stop = spec.get('start', 0), spec.get('stop', len(plan['records']))
    need(type(start) is type(stop) is int
         and 0 <= start <= stop <= len(plan['records']), 'actual panel chunk')
    return list(range(start, stop))


def require_global(c, spec, phase):
    plan = plan_document(c, spec)
    seals = {k: json.loads(read(v, 32 * MIB))
             for k, v in spec.get('phase_seals', {}).items()}
    required = {'gc': ('predict',), 'pose': ('predict', 'gc'),
                'score': ('predict', 'gc', 'pose'),
                'audit': ('predict', 'gc', 'pose')}[phase]
    for name in required:
        seal = seals[name]
        need(seal['phase'] == name and seal['plan'] == spec['plan']
             and seal['status'] == 'passed'
             and set(seal['by_pair']) == set(map(str, plan['pairs'])),
             'NEW whole-panel source/prediction/control/fit/pose seals')
        for record in seal['by_pair'].values():
            verify_payload(c, record)
    return plan, seals


def actual_weight_pins(c):
    expected = c['weights']
    need(type(expected) is list and len(expected) == 3,
         'XFeat, matched Small and sparse stock Small weight pins')
    observed = {p['path']: pin(p['path']) for p in expected}
    need(len(observed) == 3 and all(observed[p['path']] == p for p in expected),
         'exact three qualified local checkpoints')
    return observed


def absolute_preparation_pin(p):
    path = Path(p['path'])
    return dict(p, path=str(path if path.is_absolute() else PROJECT / path))


def predict(c, spec, route):
    import numpy as np
    import torch
    need(not any(k.startswith('pygcransac') for k in sys.modules),
         'neural worker has no native GC runtime')
    plan = plan_document(c, spec)
    selected = indices_for(spec, plan)
    prepared = json.loads(read(c['inputs']['prepared_manifest'], 32 * MIB))
    need(len(prepared['records']) == 1500, 'original prepared1500 metadata')
    for record in plan['records']:
        need(route.exact_json_equal(record, prepared['records'][record['pair_index']]),
             'exact original prepared directed record')
    for directory in (PROJECT / 'benchmarks/xrefine',
                      PROJECT / 'research/xfeat-star-acquisition-20261020'):
        sys.path.insert(0, str(directory))
    benchmark = load(c['seams']['benchmark'], 'benchmark')
    core = load(c['seams']['population_core'], 'core')
    image_guard = load(c['seams']['image_guard'], 'image_guard')
    preparation = load(c['seams']['preparation'], 'prepare_image_guard')
    lead = load(c['sources']['base_core'], '_rm051_existing_Small_core')
    consensus = load(c['sources']['consensus'], 'consensus051')
    data = load(c['seams']['MD_data'], 'plugbench_data')
    weight_pins = actual_weight_pins(c)
    extractor, model, processor, loaded_weights, loaded_states = preparation.load_models()
    del extractor
    for p in loaded_weights:
        absolute = absolute_preparation_pin(p)
        need(weight_pins.get(absolute['path']) == absolute,
             'original preparation actually loaded the pinned checkpoint')
    sparse_models, stock_weights = benchmark.load_models(['xrefine_small_xfeat'])
    stock_model = sparse_models['xrefine_small_xfeat'].to(torch.device('cpu')).eval()
    for parameter in stock_model.parameters():
        parameter.requires_grad_(False)
    stock_path = PROJECT / stock_weights['xrefine_small_xfeat']['path']
    need(pin(stock_path) == weight_pins.get(str(stock_path)),
         'fresh stock original checkpoint path/hash')
    model_state = lead._module_digest(model.net)
    stock_state = lead._module_digest(stock_model.net)
    guard_receipt, guarded_images = image_guard.authenticate_probe()
    preparation.GUARD_IMAGES = guarded_images
    checkpoint = pin(Path(c['root']) / 'checkpoint.json')
    outputs, witnesses = [], []
    for index in selected:
        started = time.monotonic()
        record = plan['records'][index]
        pair = record['pair_index']
        saved = prepared['records'][pair]
        values = data.load_pair(record)
        need(core.pair_signature(values) == saved['input_signature'],
             'original sparse input signature')
        bank = arrays(c, np, saved['bank'])
        core.validate_bank(bank, pair)
        for stem in ('raw', 'stock'):
            for side in (0, 1):
                need(route.array_seal(bank[f'{stem}{side}'])
                     == route.array_seal(values[stem][side]),
                     'original sparse endpoint population')
        images = preparation.processed_images(record, processor)
        for side in (0, 1):
            need(tuple(images[side].shape[-2:]) == tuple(values[f'hw{side}']),
                 'actual processed image dimensions')
        uid = route.uid('megadepth', pair)
        baseline_started = time.monotonic()
        parity = lead.real_parity(model, bank['star0'], bank['star1'], *images,
                                  bank['acquired_row_ids'], uid,
                                  expected_q=(bank['primary0'], bank['primary1']))
        baseline_seconds = time.monotonic() - baseline_started
        cache = parity.pop('cache')
        need(parity['status'] == 'BYTE_PARITY_PASS'
             and parity['saved_q_checked'] is True
             and parity['same_original_call_shapes'] is True,
             'fresh exact saved matched-Small baseline')
        p0, p1 = bank['star0'], bank['star1']
        row_ids = bank['acquired_row_ids']
        n = len(row_ids)
        edge = consensus.edge_mask(p0, p1, list(values['hw0']), list(values['hw1']))
        branches0 = np.empty((9, n, 2), np.float32)
        branches1 = np.empty_like(branches0)
        branches0[0], branches1[0] = cache['q0'], cache['q1']
        requested0 = np.repeat(p0[None], 9, axis=0)
        requested1 = np.repeat(p1[None], 9, axis=0)
        for branch, (side, axis, offset) in enumerate(SHIFTS[1:], 1):
            (requested0 if side == 0 else requested1)[branch, :, axis] += np.float32(offset)
        anchors0, anchors1 = requested0.copy(), requested1.copy()
        anchors0[:, edge], anchors1[:, edge] = p0[edge], p1[edge]
        calls = []
        for chunk in cache['chunks']:
            start, stop = chunk['start'], chunk['stop']
            calls.append(dict(branch=0, start=start, stop=stop,
                origin='existing real_parity collection; original untapped parity also executed',
                input_seals=[route.array_seal(anchors0[0, start:stop]),
                             route.array_seal(anchors1[0, start:stop])],
                requested_input_seals=[route.array_seal(requested0[0, start:stop]),
                                       route.array_seal(requested1[0, start:stop])],
                output_seals=[route.array_seal(branches0[0, start:stop]),
                              route.array_seal(branches1[0, start:stop])]))
        original_refine = preparation.point_refiner(model, images)
        branch_seconds = []
        for branch in range(1, 9):
            cursor = [0]
            def traced_refine(a0, a1):
                start, stop = cursor[0], cursor[0] + len(a0)
                need(a0.tobytes() == anchors0[branch, start:stop].tobytes()
                     and a1.tobytes() == anchors1[branch, start:stop].tobytes(),
                     'actual original-chunk effective branch anchors')
                tick = time.monotonic()
                result = original_refine(a0, a1)
                calls.append(dict(branch=branch, start=start, stop=stop,
                    input_seals=[route.array_seal(a0), route.array_seal(a1)],
                    requested_input_seals=[route.array_seal(requested0[branch, start:stop]),
                                           route.array_seal(requested1[branch, start:stop])],
                    output_seals=[route.array_seal(result[0]), route.array_seal(result[1])],
                    seconds=time.monotonic() - tick))
                cursor[0] = stop
                return result
            tick = time.monotonic()
            q0, q1 = core.batched_refine(traced_refine, anchors0[branch], anchors1[branch])
            branch_seconds.append(time.monotonic() - tick)
            need(cursor[0] == n, 'every branch executes every original row')
            for q, baseline in ((q0, branches0[0]), (q1, branches1[0])):
                need(q[edge].tobytes() == baseline[edge].tobytes(),
                     'actual unchanged-edge branch response byte parity')
                q[edge] = baseline[edge]
            branches0[branch], branches1[branch] = q0, q1
        expected_chunks = [(v['start'], v['stop']) for v in cache['chunks']]
        for branch in range(9):
            need([(v['start'], v['stop']) for v in calls if v['branch'] == branch]
                 == expected_chunks, 'all nine responses preserve original256/tail chunk boundaries')
        tick = time.monotonic()
        response = consensus.outputs(branches0, branches1, edge, row_ids, uid)
        consensus_seconds = time.monotonic() - tick
        matches = tuple(torch.from_numpy(value.copy())
                        for value in (*values['raw'], *values['descriptors']))
        tick = time.monotonic()
        with torch.inference_mode():
            stock = benchmark.refine('xrefine_small_xfeat', matches, images, sparse_models)
        stock = tuple(value.detach().contiguous().numpy().copy() for value in stock)
        stock_seconds = time.monotonic() - tick
        for side in (0, 1):
            need(stock[side].tobytes() == bank[f'stock{side}'].tobytes(),
                 'fresh original sparse stock Small byte parity')
        arm_points = [(branches0[0], branches1[0]),
            (response['C0'], response['C1']), (response['U0'], response['U1']),
            (response['M0'], response['M1']), (response['F0'], response['F1']),
            (response['S0'], response['S1']), (p0, p1), stock]
        payload = dict(branches0=branches0, branches1=branches1,
            requested_anchors0=requested0, requested_anchors1=requested1,
            anchors0=anchors0, anchors1=anchors1,
            K0=values['K0'], K1=values['K1'],
            hw0=np.asarray(values['hw0'], np.int64), hw1=np.asarray(values['hw1'], np.int64))
        for key in ('edge', 'choice', 'best', 'ungated_choice', 'fixed_choice',
                    'shuffled_choice', 'support', 'energy', 'shuffle_mapping', 'row_ids'):
            payload['response_' + key] = response[key]
        for m, points in enumerate(arm_points):
            ids = np.arange(len(points[0]), dtype=np.int64) if m == 7 else row_ids.copy()
            need(all(a.dtype == np.float32 and a.shape == (len(ids), 2)
                     and np.isfinite(a).all() for a in points),
                 'all declared arm coordinates/row populations')
            for side in (0, 1):
                payload[f'm{m}_p{side}'] = points[side]
            payload[f'm{m}_indices'] = ids
        output = save_npz(c, np, Path(plan['output']) / 'predictions' / f'{pair:04d}.npz', payload)
        need(lead._module_digest(model.net) == model_state
             and lead._module_digest(stock_model.net) == stock_state,
             'matched and stock frozen states unchanged by all calls')
        witness = dict(pair_index=pair, uid=uid, source=record, bank=saved['bank'],
            prediction=output, input_signature=saved['input_signature'],
            checkpoint=checkpoint, source_pins=dict(c['sources']), seam_pins=dict(c['seams']),
            weight_pins=list(weight_pins.values()), model_state_sha256=model_state,
            stock_model_state_sha256=stock_state, loaded_preparation_states=loaded_states,
            parity=parity, original_chunks=cache['chunks'], branch_order=BRANCH_ORDER,
            branch_calls=calls, rows=n, edge_rows=int(edge.sum()),
            edge_row_ids=row_ids[edge].tolist(),
            processed_pixel_signatures=[image_guard.pixel_signature(image) for image in images],
            processed_K=[values['K0'].tolist(), values['K1'].tolist()],
            population_counts={name: len(payload[f'm{m}_indices']) for m, name in enumerate(ARMS)},
            row_id_seals={name: route.array_seal(payload[f'm{m}_indices']) for m, name in enumerate(ARMS)},
            branch_array_seals={key: route.array_seal(payload[key]) for key in
                ('branches0', 'branches1', 'anchors0', 'anchors1', 'requested_anchors0', 'requested_anchors1')},
            call_counts=dict(distinct_response_branches=9,
                parity_model_forward_calls=parity['total_model_forward_calls'],
                parity_head_replay_calls=parity['frozen_head_replay_calls'],
                shifted_model_forward_calls=8 * len(expected_chunks),
                all_matched_model_forward_calls=parity['total_model_forward_calls'] + 8 * len(expected_chunks),
                stock_original_whole_population_calls=1 if len(stock[0]) else 0,
                no_nine_total_call_or_speed_claim=True),
            seconds=dict(baseline_parity=baseline_seconds, shifted_branches=branch_seconds,
                         consensus=consensus_seconds, stock_parity=stock_seconds,
                         whole_pair_including_IO=time.monotonic() - started),
            GT_read=False, fitted=False)
        outputs.append(output)
        witnesses.append(witness)
        print(json.dumps(dict(phase='predict', pair=pair, rows=n, edge_rows=witness['edge_rows'],
                              matched_forward_calls=witness['call_counts']['all_matched_model_forward_calls'])), flush=True)
        del cache, images, bank, values, payload, branches0, branches1, response
        del original_refine, traced_refine, arm_points, anchors0, anchors1
        del requested0, requested1, matches, stock, q0, q1, p0, p1, row_ids, edge
    return dict(outputs=outputs, witnesses=witnesses,
                actual_processed_image_guard_probe=guard_receipt,
                actual_source_bindings=dict(sources=c['sources'], seams=c['seams']),
                MD_grid_retention='one pair in memory; grids never saved',
                fitted=False, GT_read=False)


def native_phase(c, spec, route, phase):
    plan, seals = require_global(c, spec, phase)
    selected = indices_for(spec, plan)
    worker = load(c['seams']['native_worker'], '_rm051_existing_bound_worker')
    pending = set()
    def verify(p):
        verify_payload(c, p)
    def document(path):
        return json.loads(Path(path).read_bytes())
    def load_npz(np, p):
        return arrays(c, np, p)
    def emit_npz(np, path, values):
        return save_npz(c, np, path, values)
    def journal(output, stage, event, **fields):
        need(stage in ('gc', 'pose') and event in ('before', 'after')
             and fields['method'] in ARMS, 'native journal identity')
        key = (stage, tuple(sorted(fields.items())))
        if event == 'before':
            need(key not in pending, 'duplicate native before')
            pending.add(key)
        else:
            need(key in pending, 'native after without before')
            pending.remove(key)
        path = Path(output) / f'{stage}-events-{selected[0]:04d}-{selected[-1]+1:04d}.jsonl'
        body = encoded(dict(event=event, monotonic=time.monotonic(), **fields))
        need(len(body) <= 512, 'bounded native event')
        with path.open('ab') as stream:
            need(stream.write(body) == len(body), 'complete native journal append')
            stream.flush()
            os.fsync(stream.fileno())
    context = dict(Path=Path, os=os, sys=sys, ctypes=ctypes, hashlib=hashlib,
        json=json, encoded=encoded, verify=verify, document=document,
        load_npz=load_npz, save_npz=emit_npz, journal=journal)
    base, bindings = worker.bound_functions(c['seams']['base'],
        ('images_loaded', 'separation', 'verify_solver', 'gc_worker', 'recover'),
        '_rm051_original_native_base', context)
    solver = json.loads(read(c['inputs']['solver']))
    base.verify_solver(solver)
    if phase == 'gc':
        native, extra = worker.bound_functions(c['seams']['native'],
            ('require', 'descriptor', 'load_extension'), 'plugbench_native',
            dict(Path=Path, os=os, sys=sys, stat=stat, re=re,
                 hashlib=hashlib, importlib=importlib))
        sys.modules['plugbench_native'] = native
        result = base.gc_worker(plan, selected, seals['predict']['by_pair'], solver)
    else:
        sys.modules['recovery'] = route.load_module(c['seams']['recovery'], '_rm051_original_recovery')
        context = dict(Path=Path, sys=sys, json=json, need=need, base=base,
            load_npz=load_npz, save_npz=emit_npz, journal=journal,
            SMOKE=Path(plan['output']) if plan['panel'] == 'smoke' else None)
        paired, extra = worker.bound_functions(c['seams']['paired'], ('pose_worker',),
                                               '_rm051_original_paired_recovery', context)
        text = read(c['seams']['paired']).decode('utf-8')
        tree = ast.parse(text)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pose_worker')
        source = ast.get_source_segment(text, fn)
        need(source.count("len(indices)*6*plan['repeats']") == 2,
             'exact two existing role-count-only corrections')
        revised = source.replace("len(indices)*6*plan['repeats']",
                                 "len(indices)*len(plan['methods'])*plan['repeats']")
        exec(compile(revised, c['seams']['paired']['path'] + '#RM051-count-only', 'exec'), paired.__dict__)
        result = paired.pose_worker(plan, selected, seals['predict']['by_pair'], seals['gc']['by_pair'])
        result['readout_count_only_interface_extension'] = dict(
            original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
            executed_function_sha256=hashlib.sha256(revised.encode()).hexdigest(),
            replacements=2, scientific_loop_unchanged=True)
        need(result['processed_readouts'] == len(selected) * len(ARMS) * plan['repeats'],
             'actual eight-role paired readout denominator')
    need(not pending, 'all native journaled calls returned')
    result['actual_source_bindings'] = bindings + extra
    return result


def scoring_fields(c):
    text = read(c['seams']['paired']).decode('utf-8')
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score_worker')
    return {n.slice.value for n in ast.walk(fn) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == 'plan'
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}


def score(c, spec, route):
    plan, seals = require_global(c, spec, 'score')
    need(scoring_fields(c) <= set(plan), 'actual existing scorer schema including evidence_scope')
    selected = indices_for(spec, plan)
    need(selected == list(range(len(plan['records']))), 'whole globally sealed score')
    worker = load(c['seams']['native_worker'], '_rm051_existing_score_binder')
    route.load_module(c['seams']['MD_data'], 'plugbench_data')
    route.load_module(c['seams']['metrics'], 'plugbench_metrics')
    context = dict(Path=Path, io=io, ROOT=PROJECT,
        base=types.SimpleNamespace(csv=csv), load_npz=lambda np, p: arrays(c, np, p),
        document=lambda p: json.loads(Path(p).read_bytes()), encoded=encoded,
        create=lambda p, b, cap=17*MIB: create(c, p, b, cap),
        CSV_CAP=32 * MIB, pin=pin)
    module, bindings = worker.bound_functions(c['seams']['paired'], ('score_worker',),
                                             '_rm051_original_scorer', context)
    result = module.score_worker(plan, seals['predict']['by_pair'], seals['pose']['by_pair'])
    result['actual_source_bindings'] = bindings
    return result


def execute(c, spec):
    """One fresh, owned process per phase; source/resource authority stays parent-owned."""
    phase = spec['phase']
    need(phase in ('checks', 'predict', 'gc', 'pose', 'score', 'audit'), 'assigned RM051 phase')
    forbidden = ('numpy', 'torch', 'cv2', 'kornia', 'PIL', 'pygcransac')
    need(not any(name == root or name.startswith(root + '.')
                 for name in sys.modules for root in forbidden),
         'fresh worker before scientific imports')
    route = load(c['sources']['route_runtime'], '_rm051_existing_runtime')
    runtime = route.bootstrap_runtime('features' if phase in ('checks', 'predict')
                                      else 'score' if phase == 'audit' else phase)
    if phase == 'checks':
        import pytest
        load(c['sources']['consensus'], 'consensus051')
        fields = scoring_fields(c)
        need('evidence_scope' in fields, 'actual existing scorer consumer contract')
        result_code = pytest.main([c['sources']['tests']['path'], '-q', '-p', 'no:cacheprovider'])
        need(result_code == 0, 'meaningful affected consensus checks')
        result = dict(pytest_exit=int(result_code), scoring_consumer_fields=sorted(fields),
                      test_source=c['sources']['tests'])
    elif phase == 'predict':
        result = predict(c, spec, route)
    elif phase in ('gc', 'pose'):
        result = native_phase(c, spec, route, phase)
    elif phase == 'score':
        result = score(c, spec, route)
    else:
        require_global(c, spec, 'audit')
        audit = load(c['sources']['audit'], '_rm051_independent_saved_audit')
        result = audit.audit(c, spec['plan'], spec['phase_seals'])
    result['runtime'] = runtime
    return result
