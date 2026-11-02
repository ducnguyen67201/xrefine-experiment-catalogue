# Other-view endpoint consensus — prospective experiment

20 October 2026. The user requested failure analysis of the utility predictor,
a better approach following Professor Kim, implementation and the full reusable
benchmark. This separately named local experiment tests additional cached
views. It is explicitly an **n-view/transductive development experiment**; its
input contract differs from ordinary two-image XRefine. No acquisition, new
neural inference, external dispatch or publication is authorized here.

## Why change the information

The closed utility experiment scored48.21332861583971 AUC5 against fresh stock
48.40912508233161, delta-0.19579646649190607, with3/10 positive draws. It selected
RAW for29/1500 pairs. On1471 byte-identical unchanged pairs the difference was
-0.19187767998595007: unseeded native variation materially limits causal harm
attribution. Twelve RAW global summaries do not observe proposed-refinement
quality, local ambiguity or the same endpoint under another partner. Sparse
actions, weak information, two development scenes and a capped-loss target
that differs from trapezoidal AUC are plausible limitations, not proven causes.
Do not retune or reopen that gate. Preserve all prior sources, failures and
scores exactly.

Professor Kim's recorded direction was to understand failure and use relations
beyond a single local matching decision; the saved meeting interpretation is
[here](../current/PROFESSOR-KIM-DIRECTION-20261019.md). The XRefine paper itself
describes inconsistent tracks from independently refined image pairs and an
alternative second-endpoint-only n-view architecture in§3.3
([primary source](https://arxiv.org/html/2601.12530v1)). Multi-view feature
refinement is established prior art: [Dusmanu et al., ECCV2020](https://arxiv.org/abs/2003.08348)
and [PixSfM, ICCV2021](https://arxiv.org/abs/2108.08291). Our cached-coordinate
consensus is a known adaptation, not the released n-view architecture, a novel
optimizer or a comparison against those methods. Historical O was an unselected
third-image cycle-filter interface HOLD; this question keeps all matches and
supplies an explicit existing-cache n-view contract. It does not execute O or
reopen any closed question.

## Fixed information and identity contract

Use the same frozen1500 directed pairs/806 images in scenes0015 and0022 from the
previous full plan, RAW descriptors and RAW/stock endpoints only. No pose,
essential matrix, inlier mask, score, scene label or GT enters the candidate.
Exact image bytes/pins, processed K and image dimensions must be consistent
across appearances of an image. There are no detector IDs in these caches.
Therefore identify a node by exact image name, little-endian FP32 RAW xy bytes
and exact64-dimensional FP32 descriptor bytes. If the same image and RAW xy
has conflicting descriptor bytes, mark it ambiguous and fall back to stock;
never merge approximately. Count conflicts and exclusions.

For current pair(A,B), gather stock endpoint observations of that same RAW node
from all other cached partners. Exclude **every observation whose partner is
the current other image**, including reverse and repeated(A,B) observations.
Within a distinct auxiliary partner image, take the arithmetic mean of its
observations, so one partner contributes one equal vote. Retain at most8 distinct
partners ordered by SHA256 of canonical compact ASCII JSON
`[42,image,raw_xy_hex,partner]`, with partner string as collision tie breaker.
raw_xy_hex is the little-endian FP32 xy byte string. No LF in this hash wire.
All context statistics/arithmetic use FP64; cast each final output once toFP32.

A match is eligible only if **both endpoints** have at least one eligible
other-partner observation and neither identity is ambiguous. All arms use the
same eligibility mask. Unsupported rows stay byte-identical stock. Preserve
the complete correspondence count, row order and endpoint pairing; no filter,
new match, descriptor matching, GT fit or learned threshold.

## One primary and three controls

For endpoint stock u and auxiliary votes q_j, primary Huber consensus minimizes

`f(z)=.5||z-u||^2 + mean_j rho_delta(||z-q_j||)`, delta=.5 processed pixel,
where rho_delta(r)=.5r^2 for r<=delta and delta*(r-.5delta) otherwise.

Start z=u and perform exactly32 gradient steps with step size.5:
`z <- z-.5*((z-u)+mean((z-q)*min(1,.5/max(||z-q||,1e-12))))`.
The objective is1-strongly convex and its gradient has Lipschitz constant at
most2. This fixed iteration contracts at most.5, has a unique limiting solution
and limits the solution displacement to.5pixel. These mathematical properties
provide no physical-point or pose-accuracy guarantee. View-dependent useful
joint endpoint motion can be destroyed by independent endpoint averaging.

The six arms are RAW, stock XRefine small/XFeat baseline, primary `multiview_huber`,
`multiview_mean` with z=(u+mean(q))/2, `multiview_shrink` with
z=(u+floor(RAW)+.5)/2, and `multiview_shuffle`. Shuffle computes primary joint4D
delta=(source_delta,target_delta), sorts eligible rows by SHA256 of compact
ASCII JSON[42,pair_index,row_index] with row collision tie breaker, then moves
those4D blocks by one cyclic position: target sorted[i] receives source
sorted[(i-1) mod k]. Deltas are final FP32 primary endpoints minus stock,
computed in FP64. It preserves the pair's displacement
bank and eligible count while breaking row identity; singleton groups remain
identical and are reported. Mean is not displacement-capped; any primary gain
over mean can reflect bounded influence. No control can replace the primary
after scoring.

Before pose evaluation, a GT-free coverage falsifier requires in EACH scene
at least100 pairs with>=8 both-supported rows AND at least5000 supported rows.
If it fails, close this candidate as CLOSED_COVERAGE_INFEASIBLE, accuracy
UNTESTED, count this distinct unsuccessful family once, and do not claim a
full accuracy result or search another setting. Coverage counts support, not
floating-point inequality against stock. Freeze the banks before any new pose
scores. Report preparation wall time, support/conflict distribution, endpoint
dispersion and displacement, changed/no-op rows and pairs. These are cached
coordinate/precomputation costs, not neural or end-to-end inference speed.

## Full evaluation and scientific decision

After affected offline tests and an8-pair smoke, run1500 pairs x6 methods x10
native draws=90000 outcomes through the qualified compact-final-ba-v1 solver,
the unchanged GC/recovery/metric functions and the same normalization,1000GC
iterations, recovery replay and180degree failure penalty. Recompute all six
arms, retaining every draw and failure. GC and Python OpenCV remain separate.
The primary contrast is Huber minus fresh stock mean AUC5. Report AUC5/10/20,
both scenes, each draw, failed poses, >90degree severe errors, allocation,
unchanged-output subset variation and all preparation/evaluation costs.

Pass requires primary mean delta>=.5AUC5 points; positive mean in both scenes;
positive delta in>=8/10draws; strictly higher mean than mean, shrink and shuffle;
no increase in failed poses or severe errors versus stock; and actual changed
outputs in both scenes. This is a conjunction, fixed before numerical reads.
Failure closes this exact candidate; do not tune thresholds, discard failures,
promote a control or infer independent-scene generalization. Even a development
pass does not establish superiority under the standard two-image input budget.

## Finite storage, preservation and execution

New finite namespace research/multiview-consensus-20261020; new smoke/full
results under the corresponding results/xrefine names; benchmark additions
multiview_consensus_core.py, test_multiview_consensus.py, four explicit
plugbench_multiview adapters, plugbench_multiview_transport.py and its test.
Additional namespace sources prepare.py, audit.py, execute.py, postflight.py and
the metadata-only freeze.py publisher
are source-reviewed before use. Each source<=128KiB; all new source<=256KiB.
New physical peak across these roots/sources/staging<=80files, metadata/source<=64MiB
(excluding bank ZIP payloads and declared numerical event journals, which still
count fully toward aggregate),
aggregate campaign<=1GiB; metadata file<=16MiB, NPZ payload<=32MiB, ONLY this
new transport's ZIP containers<=128MiB. Old64MiB transport rules are unchanged.
Three prepared banks and nine full numerical phase containers have embedded
member indices. New smoke, observer and phase evidence may be archived
losslessly after closure, with exact byte/member hashes and readback before
removing their new originals. No historical file is deleted, repacked or edited.
The full old global union must fit its unchanged10000file and byte limits,
including4reserved files and the entire80file future envelope, BEFORE launch.

Original prediction, GC, recovery, scoring and process-supervision function
bodies remain unchanged and pinned. The separately named transport changes
storage, snapshot/receipt reads and phase orchestration only. Preserve exact
inner NPZ bytes and all journal/log/seal/receipt/source-snapshot bytes with
duplicate/path/tamper/member-limit checks. New journal aggregate cap256MiB and
individual chunk journal<=64MiB, checked before append. No silent loss of
native event evidence. Partial failures are retained and not scored as success.

ROOT and three peers prospectively review this design and all saved source,
inputs, exact commands and freeze before numerical execution. ROOT alone runs
serial managed stages: offline<=180s, prepare<=600s, smoke<=300s, full<=9000s,
inclusive allocation10080s. Existing charge34563.44731316697s/86400, committed0,
failed families5/6; no reset or implicit new allowance. One unsuccessful
scientific family is counted once; no action after the6family ceiling. Each
worker<=3600s, four threads,8GiB sampled combined RSS, original observer kill/
reap and dual locks, fresh ordinary-usage proof<=300s,>=8GiB free/inactive RAM,
>=32GiB disk and no conflicting project heavy processes. Reserve before work,
settle only after verified process/group closure, use an inclusive30s settlement
reserve and deadline90s tail. A failed stage has no automatic retry. All prior
branches and ledger bytes stay exact; new state is a separate branch and plan
entries are append-only. Automation remains PAUSED. A final independent audit
checks pins, outcomes, decision, costs, preservation and idle owner. Current
docs are derived after settlement; they confer no execution authority.
