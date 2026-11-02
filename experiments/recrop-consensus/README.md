# Recropping and local response consensus

I tested whether another observation around a refined position improves the refinement.

**Capture diagnostic (19 October).** The same frozen11×11 Small model runs again after recentering at its first outputs. Raw, one-pass, repeated-original, extrapolation and phase controls separate extra compute and rounding effects. In2048 controlled identity-warp cases, mean alignment error fell0.433→0.109px, but already-aligned anchors moved too far and the fixed stability gate failed. This was not a camera-pose improvement.

**RM051 response consensus.** Nine real responses comprise baseline and ±1-pixel recrops at either endpoint. A joint medoid needs support from at least5/9 responses, else exact baseline. Same-compute alternatives and branch shuffle are controls. Pilot AUC5: baseline51.8040 → candidate51.6717; only4/10 draws positive. Severe4→2, but no causal tail improvement is established. The exact variant closed without a full run.

**Learning.** Local agreement can be stable yet incorrect. Better capture on an identity task does not establish pose-helpful localization.

The RM051 scientific core/worker are the original evaluated versions. Later relative-path auditor repairs changed operational provenance only; those wrappers are omitted.

## Handpicked files

| File | Purpose |
|---|---|
| [capture-20261019/actor-01.py](capture-20261019/actor-01.py) | Original crop/refine actor |
| [capture-20261019/checker-01.py](capture-20261019/checker-01.py) | Independent alignment/stability checker |
| [capture-20261019/protocol.md](capture-20261019/protocol.md) | Frozen protocol |
| [response-rm051/consensus051.py](response-rm051/consensus051.py) | Joint-endpoint response consensus |
| [response-rm051/worker051.py](response-rm051/worker051.py) | Actual recrop generation and pipeline integration |
| [response-rm051/test_consensus051.py](response-rm051/test_consensus051.py) | Checks |
| [response-rm051/PROTOCOL.md](response-rm051/PROTOCOL.md) | Frozen protocol |

Exact original locations, source bindings and hashes are in [PROVENANCE.json](../../PROVENANCE.json). [Back to catalogue](../../README.md).
