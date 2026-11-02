"""One fixed attention aggregation variant over the unchanged RM042 decoder.

The immutable base supplies capture, normalization, graph, loss and optimizer
semantics. Overrides below are the entire prospective model change.
"""
from pathlib import Path
import hashlib

BASE = Path(__file__).parents[1]/'crossmatch-042/crossmatch_042.py'
body = BASE.read_bytes()
if hashlib.sha256(body).hexdigest() != '1756901882a6bb094c46f63d4697dc9c7870201e6a95d70b350d39214d449e09':
    raise RuntimeError('immutable base core digest')
exec(compile(body, str(BASE), 'exec', dont_inherit=True), globals())

PARAMETERS = 79520
_original_decode = _decode_with_context
_original_trainer = train_heads


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, dropout=0., batch_first=True)
        self.rank_embedding = nn.Parameter(torch.empty(8, 128))
        nn.init.normal_(self.rank_embedding, std=.02)
        self.layers = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 128))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        _need(sum(p.numel() for p in self.parameters()) == PARAMETERS,
              'identical 79520-parameter attention heads')

    def forward(self, value):
        return .1*torch.tanh(self.layers(value))

    def attend(self, g, indices):
        n = g.shape[0]
        _tensor(g, (n, 128), 'attention query features')
        _need(torch.is_tensor(indices) and indices.dtype == torch.int64
              and indices.device.type == 'cpu' and indices.ndim == 2
              and indices.shape[0] == n and 0 <= indices.shape[1] <= 8,
              'actual bounded token index matrix')
        k = indices.shape[1]
        _need(not k or bool(((indices >= 0) & (indices < n)).all()), 'token index bounds')
        if not n or not k:
            return torch.zeros_like(g)
        chunks = []
        for start in range(0, n, QUERY_CHUNK):
            stop = min(start+QUERY_CHUNK, n)
            tokens = g[indices[start:stop]]+self.rank_embedding[None, :k]
            value, _ = self.attention(g[start:stop, None], tokens, tokens, need_weights=False)
            chunks.append(value[:, 0])
        result = torch.cat(chunks)
        _tensor(result, (n, 128), 'finite learned attention context')
        return result


def context(g, neighbors, arm, row_ids, pair_uid):
    n = len(row_ids)
    _tensor(g, (n, 128), 'standardized evidence')
    _ids(row_ids, n)
    _uid(pair_uid)
    _need(arm in ARMS, 'declared attention control')
    k = min(8, max(n-1, 0))
    if arm == 'independent':
        return torch.arange(n, dtype=torch.int64)[:, None].repeat(1, k)
    _neighbors(neighbors, row_ids)
    indices = (shuffled_neighbors(neighbors, row_ids, pair_uid)
               if arm == 'shuffled' else neighbors)
    return torch.from_numpy(indices.copy())


def _decode_with_context(net, cache, adapter, g, indices, provenance):
    _need(isinstance(adapter, Adapter), 'actual attention head')
    return _original_decode(net, cache, adapter, g, adapter.attend(g, indices), provenance)


def _prepare_training(cache, normalizer):
    n = len(cache['row_ids'])
    g = _standardized(cache, normalizer)
    neighbors = build_neighbors(cache['geometry'], cache['row_ids'])
    messages = shuffled_neighbors(neighbors, cache['row_ids'], cache['pair_uid'])
    empty = np.empty((n, 0), np.int64)
    contexts = {arm: context(g, neighbors, arm, cache['row_ids'], cache['pair_uid'])
                for arm in ARMS}
    provenance = dict(context=_provenance(cache, 'context', neighbors, neighbors),
        independent=_provenance(cache, 'independent', empty, empty),
        shuffled=_provenance(cache, 'shuffled', neighbors, messages))
    return dict(g=g, contexts=contexts, provenance=provenance)


def _prepared_bytes(prepared):
    tensors = sum(t.numel()*t.element_size() for e in prepared.values()
                  for t in (e['g'], *e['contexts'].values()))
    arrays = sum(v.nbytes for e in prepared.values() for p in e['provenance'].values()
                 for v in p.values() if type(v) is np.ndarray)
    return int(tensors), int(arrays)


def train_heads(net, fit_caches, targets):
    result = _original_trainer(net, fit_caches, targets)
    result['metadata'].update(variant='RM045_LEARNED_CROSS_MATCH_ATTENTION',
        attention_heads=4, attention_dim=128, rank_tokens=8, dropout=0.,
        contexts_prepared_once=False, token_indices_prepared_once=True,
        independent_context_aliases_g=False,
        prepared_memory_scope='pooled g and three int64 token matrices plus actual provenance; attention contexts recomputed differentiably each update')
    return result
