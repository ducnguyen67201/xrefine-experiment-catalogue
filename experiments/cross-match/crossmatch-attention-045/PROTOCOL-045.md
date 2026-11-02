# RM045: one learned cross-match attention variant

Status: PROPOSED, before model implementation or training. User selected this
single variant in the current master chat on 29 October 2026. No other method,
research chat, worker, scheduler, model setting or worktree is started.

The prior RM042 eight-neighbor mean/FiLM variant is CLOSED_FIXED_NEGATIVE:
MD32x10 B52.1839966565 versus C44.8857070775 AUC5. It is not retuned here.
RM045 changes the aggregation to learned attention; originality is unestablished.

## Model and controls frozen before learning

Existing: same original acquired rows, frozen STAR-matched XRefine Small,
11x11 patches, tapped 64x3x3 features per endpoint, 128 pooled standardized
features per match, and whole-pair eight-nearest-neighbor 4D coordinate graph.
Reuse the authenticated ALL300 RM042 feature seal and every immutable cache;
no new features, altered row populations or unsealed label reads.

Proposed trainable block: four-head scaled dot-product attention, embed_dim128,
dropout0, with the current match as query and at most eight message tokens as
keys/values. Eight learned 128D rank embeddings distinguish token slots.
The attention result and the query enter the same 256->32->128 FiLM MLP.
Gamma=0.1*tanh(MLP), last MLP projection initially zero, modulates the same
captured feature channels before the frozen original spatial decoder.
All three heads have 79520 parameters: attention66048 + rank1024 + FiLM12448.
Each head has its own optimizer; identical initial weights and data ordering.

C uses the actual other-match neighbor tokens. Independent uses its own
feature repeated in the same number of token slots plus the same trainable
rank embeddings, so its attention remains trainable while it cannot read
other-match features. Shuffled uses the existing fixed query-specific
other-ID derangement and the same rank embeddings. Degree, dimensions,
parameter count, initialization and optimization are identical. No learned
filtering, match deletion, solver selection or confidence thresholds.
For N<=1 no other-match message exists; all arms return zero context. For
2<=N<=9 C/S may share the complete other set; rank embeddings can still make
their token-order-dependent attention differ. No permutation-invariance claim.

Attention formula per head: softmax((Wq*g_i)*(Wk*(g_j+e_rank))^T/sqrt(32))
times Wv*(g_j+e_rank), concatenated and projected to128. Query computations
may be chunked; the candidate set always comes from the entire original pair.

## Training and evidence boundaries

Same FIT150 rooms0707-0716 / 110245 acquired rows; same already-exposed DEV150
rooms0717-0726. ALL300 features/source/rows must reauthenticate before FIT GT.
No DEV-label, MegaDepth0015/0022, or held FINAL1200 fitting. Same weak calibrated
pixel-Sampson objective and displacement/gamma regularizers as042, not pose
utility. Same seed20261027, 20 epochs, LAST only, AdamW lr0.001/betas0.9,0.999/
eps1e-8/weight_decay1e-4, gradient norm1.0. No score-based tuning or best epoch.
Weights/optimizer states have a prospective4MiB checkpoint bound (larger head),
within the same finite experiment retention envelope; no old cap is rewritten.

## Pose comparison and decision frozen before scores

Same six arms B/C/independent/shuffled/nativeSTAR/stock and same qualified
acquired-bank pipeline: corrected1024/K/RGB, GC1000, OpenCV conditional recovery,
all rows, failures180, global prediction/GC/pose seals before score GT.
Engineering smoke: original8x6x1 plus original forced recovery witnesses.
Pilot: same original metadata-selected32 IDs (16/scene), ten fresh repeats.
Pass requires C-B AUC5>=0.5pp, each scene positive, C>I and C>S globally,
>=8/10 strict-positive C-B draws, no global failure or >90deg severe increase,
and all mechanical/provenance/row/GT/engineering checks.
Valid negative closes this variant; no threshold, seed, epoch or control promotion.
Only a pilot pass qualifies the full original1500x10 six-arm comparison, subject
to actual remaining resources/runtime forecast. Full adds per-scene failure and
severe nonincrease. Independent confirmation requires a frozen full pass and
actual affordability. A pilot gain is never reported as a full/main gain.
Same independent scalar reduction and fixed10000x10 Random20261027 repeat matrix,
descriptive conditional fixed-panel solver-repeat intervals, reporting only.

## Qualification and execution

One current-chat owner in the existing51c8 checkout. Checkpoint named owned
source/tests/config before execution. Meaningful affected checks: independent
scalar attention agreement, live q/k gradients, independent feature isolation,
relation-shuffle provenance, capacity/init equality, neutral decoder parity,
full-pair graph across chunk boundaries, finite/all-row outputs and immutable
base state. Reuse existing49 unaffected core checks; do not relabel them as new.
Original RM045 repair limits: max2 guard and max3 crash corrections after initial,
persistent across all its stages. Old042/039/041 failures remain closed history.
Use stable shared heavy lock dev16777231/ino204740392 through actual closure,
CPU4/RSS8GiB/free-inactiveRAM8GiB/defaultDISK32GiB + all prior held liabilities.
No installs, remote/paid jobs, protected canonical writes, destructive cleanup,
or final-data selection. Keep current source/Git/cache/failed/raw/model/audit
liabilities, not free-space credit. Actual hard137100 campaign bound remains;
known prior covered129254.37229020586508947 is incomplete and not spendable.
New local envelope is at most3500 enclosing job seconds including cleanup,
within the prior conservative remaining4072.77048541423982106 before unmeasured
overhead. Per-process hang protection and tail240 apply; actual runtime and
remaining capacity are rechecked before stages. Full runtime is UNKNOWN until
admitted smoke/pilot. Existing feature caches are read-only retained inputs.

## Primary comparison and falsifier

Known adaptation of multi-head attention (Vaswani et al.,2017), applied to
fixed-match refinement rather than assignment. SuperGlue uses attention for
matching/filtering; this study preserves acquisition and changes only spatial
refinement. Primary sources: https://arxiv.org/abs/1706.03762 and
https://github.com/magicleap/SuperGluePretrainedNetwork . No novelty claim.
Counterexample: nearby rows can belong to incompatible structures or repeat a
wrong pattern; learned attention can amplify that error. The fixed controlled
pose pilot falsifies useful benefit for this implementation. Weak-loss success
alone does not establish improved pose accuracy or cross-domain generalization.
