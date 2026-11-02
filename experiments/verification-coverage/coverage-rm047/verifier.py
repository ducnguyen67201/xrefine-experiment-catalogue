"""Frozen RM047 native-input distinctiveness and joint-cell coverage policy.

The native acquisition is observed, never replaced. No GT, coordinate edits,
scale resampling, learned parameter, threshold search or solver intervention.
"""
import math

ARMS = ['star_small', 'coverage_ambiguity', 'confidence_only', 'equal_random',
        'coverage_shuffled', 'coverage_random', 'aligned_topk',
        'coverage_confidence', 'star_native', 'stock_small']
EPS = 1e-12

def need(value, message):
    if not value:
        raise ValueError(message)

def retained_count(n):
    return min(n, max(5, (n + 1) // 2))

def selected(np, score, k):
    need(score.ndim == 1 and np.isfinite(score).all() and 0 <= k <= len(score), 'finite rank')
    return np.sort(np.lexsort((np.arange(len(score)), -score))[:k]).astype(np.int64)

def coverage(np, p0, p1, hw0, hw1, k):
    n = len(p0)
    need(p0.shape == p1.shape == (n, 2) and np.isfinite(p0).all() and np.isfinite(p1).all(), 'raw finite endpoints')
    need(0 <= k <= n and (n == 0 or k > 0), 'coverage count')
    raw = np.column_stack((p0 / np.asarray(hw0[::-1]), p1 / np.asarray(hw1[::-1]))).astype(np.float64)
    outside = ((raw < 0) | (raw >= 1)).any(axis=1)
    for g in (4, 2, 1):
        labels = np.clip(np.floor(raw * g), 0, g - 1).astype(np.int64)
        cells = ((labels[:, 0] * g + labels[:, 1]) * g + labels[:, 2]) * g + labels[:, 3]
        occupied, counts = np.unique(cells, return_counts=True)
        if len(occupied) <= k:
            break
    c = len(occupied); quotas = np.ones(c, dtype=np.int64)
    remain, denom = k - c, n - c
    if denom:
        divisions = [divmod(remain * (int(v) - 1), denom) for v in counts]
        quotas += np.asarray([x[0] for x in divisions], dtype=np.int64)
        leftover = k - int(quotas.sum())
        order = sorted(range(c), key=lambda i: (-divisions[i][1], int(occupied[i])))
        for i in order[:leftover]:
            quotas[i] += 1
    need(int(quotas.sum()) == k and bool(np.all(quotas <= counts)), 'exact capacity quotas')
    return dict(grid=np.asarray(g, dtype=np.int64), cells=cells, labels=labels,
                occupied=occupied, quotas=quotas, outside=outside)

def covered_selected(np, score, policy):
    ids = []
    need(np.isfinite(score).all(), 'finite coverage score')
    for cell, q in zip(policy['occupied'], policy['quotas']):
        rows = np.flatnonzero(policy['cells'] == cell)
        ids.extend(rows[np.lexsort((rows, -score[rows]))[:int(q)]].tolist())
    return np.asarray(sorted(ids), dtype=np.int64)

def normalized_margin(torch, fine):
    need(fine.ndim == 2 and fine.shape[1] == 128 and torch.isfinite(fine).all().item(), 'finite actual128D fine input')
    halves = []
    for f in fine.split(64, dim=1):
        halves.append(f / torch.linalg.vector_norm(f, dim=1, keepdim=True).clamp(min=EPS))
    a, b = halves
    if len(a) < 2:
        return torch.zeros(len(a), dtype=a.dtype), halves
    sim = a @ b.T
    own = sim.diag().clone(); sim.fill_diagonal_(-float('inf'))
    return torch.minimum(own - sim.max(dim=1).values, own - sim.max(dim=0).values), halves

def scale_fixtures(torch, extractor):
    """Actual vendor preprocessing/soft-offset calls on analytic non-data fixtures."""
    import torch.nn.functional as F
    rows = []
    for h, w in ((101, 67), (129, 193)):
        for scale in (.6, 1.3):
            image = F.interpolate(torch.zeros(1, 3, h, w), scale_factor=scale, mode='bilinear', align_corners=False)
            sh, sw = image.shape[-2:]; processed, sy, sx = extractor.preprocess_tensor(image)
            ph, pw = processed.shape[-2:]
            need((sh, sw) == (math.floor(h * scale), math.floor(w * scale)) and
                 (ph, pw) == (sh // 32 * 32, sw // 32 * 32), 'analytic actual scale/preprocess dimensions')
            coarse = torch.tensor([[8., 16.]]) * torch.tensor([sx, sy]) / scale
            expected = torch.tensor([[8 * sw / pw / scale, 16 * sh / ph / scale]])
            need(torch.allclose(coarse, expected, atol=1e-5, rtol=0), 'vendor nominal-scale fixture')
            rows.append(dict(original_hw=[h, w], scale=scale, actual_scaled_hw=[sh, sw], preprocessed_hw=[ph, pw],
                             coarse_native=coarse.tolist(), no_actual_dimension_or_pixel_center_correction=True))
    need(abs(rows[0]['coarse_native'][0][0] - 16.6666666667) < 1e-5, 'noninteger resize hand witness')
    logits = torch.zeros(1, 64); logits[0, 2 * 8 + 5] = 20
    offset = extractor.subpix_softmax2d(logits.view(-1, 8, 8))
    need(torch.allclose(offset, torch.tensor([[1., -2.]]), atol=1e-6, rtol=0), 'actual vendor xy/temperature fine-offset fixture')
    return dict(status='passed', scale_rows=rows, native_offset=offset.tolist(), GT_used=False, neural_forward=False)

def native_capture(torch, np, extractor, images, expected):
    """Observe original dense, dualscale, coarse gather, fine input and logits.

    Instance wrappers delegate to the original methods without changing inputs
    or results. Every tensor witness is cloned before native in-place offsets.
    Both forward hooks explicitly return None and are removed in finally.
    """
    names = ('extractDense', 'extract_dualscale', 'preprocess_tensor', 'refine_matches')
    original = {n: getattr(extractor, n) for n in names}
    missing = object(); attributes = {n: extractor.__dict__.get(n, missing) for n in names}
    dense, dual, fine = [], [], []; active = [None]; pre = {}
    def clone(t):
        return t.detach().clone()
    def prep(x):
        out = original['preprocess_tensor'](x)
        if active[0] is not None:
            d = dense[active[0]]
            need('preprocess' not in d, 'one actual preprocessing per dense extraction')
            d['preprocess'] = [*x.shape[-2:], *out[0].shape[-2:], float(out[1]), float(out[2])]
        return out
    def net_hook(module, args, out):
        if active[0] is not None:
            d = dense[active[0]]; need('map' not in d, 'one actual dense network forward')
            d['map'], d['reliability'] = clone(out[0]), clone(out[2])
        return None
    def fine_hook(module, args, out):
        fine.append((clone(args[0]), clone(out)))
        return None
    def dense_call(x, top_k=8000):
        i = len(dense); dense.append(dict(requested=int(top_k))); active[0] = i
        try:
            out = original['extractDense'](x, top_k)
            dense[i]['keypoints'], dense[i]['descriptors'] = map(clone, out)
            return out
        finally:
            active[0] = None
    def dual_call(x, top_k, s1=.6, s2=1.3):
        start = len(dense); out = original['extract_dualscale'](x, top_k, s1, s2)
        dual.append(dict(start=start, stop=len(dense), scales=[s1, s2], input_hw=list(x.shape[-2:]),
                         keypoints=clone(out[0]), scale_tags=clone(out[1]), descriptors=clone(out[2])))
        return out
    def refine(d0, d1, matches, batch_idx, fine_conf=.25):
        need(batch_idx == 0 and fine_conf == .25 and not pre, 'one unchanged original refinement')
        i0, i1 = matches[batch_idx]
        pre.update(i0=clone(i0), i1=clone(i1), coarse0=clone(d0['keypoints'][batch_idx][i0]),
                   coarse1=clone(d1['keypoints'][batch_idx][i1]), scale0=clone(d0['scales'][batch_idx][i0]),
                   scale1=clone(d1['scales'][batch_idx][i1]))
        return original['refine_matches'](d0, d1, matches, batch_idx, fine_conf)
    handles = []
    try:
        extractor.extractDense = dense_call; extractor.extract_dualscale = dual_call
        extractor.preprocess_tensor = prep; extractor.refine_matches = refine
        handles.append(extractor.net.register_forward_hook(net_hook))
        handles.append(extractor.net.fine_matcher.register_forward_hook(fine_hook))
        with torch.inference_mode():
            fresh = extractor.match_xfeat_star(images[0][None], images[1][None], top_k=10000)
    finally:
        for h in handles:
            h.remove()
        for name, old in attributes.items():
            if old is missing:
                delattr(extractor, name)
            else:
                setattr(extractor, name, old)
    need(len(dense) == 4 and len(dual) == 2 and len(fine) == 1, 'complete native dualscale/fine lineage')
    need(all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(fresh, expected)), 'EXACT final native endpoint/order/dtype byte parity BEFORE row binding')
    inputs, logits = fine[0]
    need(torch.isfinite(inputs).all().item() and torch.isfinite(logits).all().item(), 'finite native witness')
    # One pre-filter reduction/mask is shared by EVERY survivor witness.
    confidence = torch.softmax(logits * 3, dim=-1).max(dim=-1).values
    mask = confidence > .25
    need(len(inputs) == len(pre['i0']) == len(logits) and int(mask.sum()) == len(expected[0]), 'single survivor mask row binding')
    arrays = {}
    def save(name, value):
        arrays[name] = value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value)
    for i, d in enumerate(dense):
        f = d['map'].permute(0, 2, 3, 1).reshape(1, -1, 64)
        rel = d['reliability'].reshape(1, -1)
        count = min(f.shape[1], d['requested'])
        ix = torch.topk(rel, k=count, dim=-1).indices
        need(torch.equal(torch.gather(f, 1, ix[..., None].expand(-1, -1, 64)), d['descriptors']), 'actual dense descriptor gather')
        h, w = d['map'].shape[-2:]
        xy = torch.stack((ix % w, ix // w), dim=-1) * 8
        point = xy * torch.tensor([d['preprocess'][5], d['preprocess'][4]])
        need(torch.equal(point, d['keypoints']), 'actual coarsegrid/preprocess coordinate replay')
        for key, val in [('grid_features', f[0]), ('reliability', rel[0]), ('topk', ix[0]),
                         ('keypoints', d['keypoints'][0]), ('descriptors', d['descriptors'][0]),
                         ('preprocess', d['preprocess']), ('grid_hw', [h, w]), ('requested', d['requested'])]:
            save(f'l_dense_{i}_{key}', val)
    for side, d in enumerate(dual):
        need((d['start'], d['stop']) == (side * 2, side * 2 + 2) and d['scales'] == [.6, 1.3], 'actual native0.6/1.3 extraction, no added scale minimum')
        ds = dense[side * 2:side * 2 + 2]
        need(torch.equal(torch.cat([ds[j]['keypoints'] / d['scales'][j] for j in range(2)], 1), d['keypoints']), 'native nominal-scale coordinate concatenation')
        need(torch.equal(torch.cat([x['descriptors'] for x in ds], 1), d['descriptors']), 'native descriptor concatenation')
        tags = torch.cat([torch.ones_like(ds[j]['keypoints'][..., 0]) * (1 / d['scales'][j]) for j in range(2)], 1)
        need(torch.equal(tags, d['scale_tags']), 'full BOTH-side native scale-tag lineage')
        need(torch.equal(pre[f'scale{side}'], d['scale_tags'][0][pre[f'i{side}']]), 'BOTH pre-refinement selected scale-tag joins')
        for key in ('keypoints', 'scale_tags', 'descriptors', 'input_hw', 'scales'):
            value = d[key][0] if key in ('keypoints', 'scale_tags', 'descriptors') else d[key]
            save(f'l_dual_{side}_{key}', value)
    need(torch.equal(inputs[:, :64], dual[0]['descriptors'][0][pre['i0']]) and
         torch.equal(inputs[:, 64:], dual[1]['descriptors'][0][pre['i1']]), 'actual128D fine-input halves exact selected coarse descriptors')
    offsets = extractor.subpix_softmax2d(logits.view(-1, 8, 8))
    replay0 = pre['coarse0'] + offsets * pre['scale0'][:, None]
    need(replay0[mask].numpy().tobytes() == expected[0].tobytes() and pre['coarse1'][mask].numpy().tobytes() == expected[1].tobytes(), 'native fineoffset/scalecoordinate final replay')
    for key, value in pre.items():
        save('l_pre_' + key, value)
    for key, value in [('fine_inputs', inputs), ('fine_logits', logits), ('native_confidence_pre', confidence),
                       ('survivor_mask', mask), ('fine_offsets', offsets)]:
        save('l_' + key, value)
    survivor = inputs[mask]
    with torch.inference_mode():
        scores, halves = normalized_margin(torch, survivor)
    for key, value in [('v_scores', scores), ('v_feature0', halves[0]), ('v_feature1', halves[1]),
                       ('v_confidence', confidence[mask]), ('v_survivor_pre_ids', torch.nonzero(mask, as_tuple=False).flatten())]:
        save(key, value)
    return arrays

def predictions(np, bank, capture, record):
    n = len(bank['star0']); k = retained_count(n)
    score, conf = capture['v_scores'], capture['v_confidence']
    need(score.shape == conf.shape == (n,) and np.isfinite(score).all(), 'all survivor scores')
    policy = coverage(np, bank['star0'], bank['star1'], record['hw0'], record['hw1'], k)
    rng = np.random.default_rng(20261030 + 1009 * int(record['pair_index']))
    random_perm = rng.permutation(n); score_perm = rng.permutation(n); priorities = rng.random(n)
    shuffled = score[score_perm]
    ids = [np.arange(n, dtype=np.int64), covered_selected(np, score, policy), selected(np, conf, k),
           np.sort(random_perm[:k]).astype(np.int64), covered_selected(np, shuffled, policy),
           covered_selected(np, priorities, policy), selected(np, score, k), covered_selected(np, conf, policy),
           np.arange(n, dtype=np.int64), np.arange(len(bank['stock0']), dtype=np.int64)]
    arrays = dict(capture, K0=np.asarray(record['K0'], dtype=np.float64), K1=np.asarray(record['K1'], dtype=np.float64),
                  hw0=np.asarray(record['hw0'], dtype=np.int64), hw1=np.asarray(record['hw1'], dtype=np.int64),
                  v_all_row_ids=bank['acquired_row_ids'].copy(), v_random_permutation=random_perm,
                  v_score_permutation=score_perm, v_random_priorities=priorities, v_shuffled_score=shuffled)
    arrays.update({'v_' + key: val for key, val in policy.items()})
    for m, ix in enumerate(ids):
        stem = 'star' if m == 8 else 'stock' if m == 9 else 'primary'
        for side in (0, 1):
            arrays[f'm{m}_p{side}'] = bank[f'{stem}{side}'][ix].copy()
        arrays[f'm{m}_indices'] = ix
    need(all(len(ids[m]) == k for m in range(1, 8)), 'seven equal-count filtered arms')
    return arrays
