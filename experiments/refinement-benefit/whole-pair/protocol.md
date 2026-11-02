# Joint/raw refinement utility — prospective experiment

20 October 2026, America/Toronto. The user explicitly requested analysis of the
fixed-source negative, a materially different candidate following Professor Kim,
implementation and evaluation through the reusable benchmark, and a standing
instruction that implemented research candidates receive that evaluation.

## Preserved result and hypothesis

The complete previous development comparison measured stock joint AUC5
48.48646762663099 versus fixed-source 43.00181659920148: **−5.484651027429507
points**, negative in ten draws and both scenes. Centered-source also lost
5.420452875786445 points. Preserve all original sources, results and failed
engineering attempts; do not rename the fixed-source version, tune it or change
its descriptive campaign accounting. Record its readout as unfavorable/closed
for promotion. The user's shorthand “5.0” does not create a software version.

The XRefine source jointly updates both patch endpoints. For frozen reference
geometry F, joint residual is v^T F u and fixed-source residual v^T F a; their
exact difference is v^T F(a-u). Dropping source motion can remove a jointly
useful correction. This is an explanatory hypothesis, not proof of physical
point correctness or the cause of each failed pose. Pixel centering alone did
not restore joint accuracy. [XRefine §§3–4](https://arxiv.org/html/2601.12530v1)
and local model.py/benchmark.py corroborate the joint versus distinct
second-keypoint-only architectures. Generic adaptive confidence computation is
known, e.g. [LightGlue](https://openaccess.thecvf.com/content/ICCV2023/papers/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.pdf).

Test one new question: **Can a pre-refinement, cross-match summary predict the
whole image pair's pose utility of stock joint refinement versus retaining raw
coordinates?** The binary action keeps both endpoints and all correspondences
together. It does not filter matches, project points, correct a learned residual
or retry closed AR, recrop, Sampson-projection or nested residual-gate versions.
It is a known utility-regression adaptation, with originality unestablished.

## Fixed feature, target and model definitions

Use the accepted previous full 75,000-row CSV only for descriptive failure
analysis and training labels. For pair i define

`y_i = mean_draw [min(error_raw,5)-min(error_stock,5)] / 5`.

Positive y favors joint refinement. This bounded expected capped-pose-loss
utility is a proxy for the reported trapezoidal AUC5, not an identical metric
or a per-match correctness label. Pose failures retain 180 degrees. Report
whole-pair GT-assisted selection headroom clearly as a diagnostic, never a
deployable selector or exact AUC upper bound.

The candidate uses twelve fixed RAW-input features, in this order:

1. log(1+number of matches).
2. Descriptor-cosine 10th, 50th and 90th percentiles.
3. Descriptor-cosine mean and population SD.
4. Source and target 4×4 grid occupancy fraction, using their own nonzero
   bounding-box extent; constant axes use extent 1 and inputs are retained.
5. Source and target raw-pixel population-covariance minimum/maximum eigenvalue ratios
   (zero for zero maximum eigenvalue; clip numerical ratios to[0,1]).
6. Median and 90th percentile affine-fit residual: inverse-K normalized raw
   rays, least-squares source-to-target affine map including intercept, Euclidean
   residual divided by max(target RMS spread,1e-8). Fewer than eight rows or
   rank-deficient design uses residual features 0 and 0. No robust fitting,
   RANSAC, reference geometry, stock movement, pose score or scene ID is a
   numerical feature. Empty/degenerate fixtures must remain finite.

Compute cosine from L2-normalized descriptor rows (norm floor 1e-12), clip to
[-1,1]. Use float64 feature/training arithmetic. For each held-out scene, train
ONLY on the other scene. Center and scale each feature with training mean and
population SD floored at 1e-6; clip standardized values to [-10,10] for both
training and inference. Recenter clipped Z using ONLY its training column means: C=Z-mean_train(Z),
for both training and inference. Fit an unpenalized mean-utility intercept plus
ridge coefficients `(C^T C + 10 I)^-1 C^T (y-mean(y))`, predicting
`mean_train(y)+C_test*coefficients`. This minimizes summed squared training
residuals plus10 times squared coefficient norm; the intercept is unpenalized.
There is one fixed L2 value,
no search, target adjustment, threshold fitting or model selection after scores.
Choose stock if predicted utility >=0, otherwise raw. No fractional movement.

The two folds are train0015/evaluate0022 and train0022/evaluate0015. Verify
disjoint scene/image inventories. Construct separate training-label projections
for each fold; feature scaling, labels, parameters and thresholds must not use
that fold's held-out labels. Percentiles use linear interpolation; covariance is population. Feature names are fixed above. Held-out features
may be computed without labels to produce predictions/control counts. These
are two exposed development scenes; this OOF experiment is NOT independent
final-scene confirmation. The adapter is an explicit research OOF model, not a
validated universal deployment model. Save fold training identities and label
projection hashes separately from inference-only checkpoints.

## Locked full comparison and decision

Run 1,500 directed pairs / 806 images, ten native draws and five arms:

- `raw`: always retain raw; `xrefine_small_xfeat`: always joint (baseline).
- `utility_gate`: twelve-feature cross-scene predictor (primary candidate).
- `confidence_gate`: the same ridge/folds/target using ONLY mean descriptor
  cosine (feature index4); tests added cross-match information against a cheap
  confidence explanation.
- `shuffled_gate`: within each held-out scene retain the primary's raw-choice
  COUNT, assigning those raw choices by ascending SHA256 of seed42 plus exact
  ordered image names, encoded as canonical compact ASCII JSON `[42,name0,name1]` plus one LF; use pair_index only as a collision tie-breaker. It uses no held-out labels and retains every match.

Primary contrast is utility_gate minus stock AUC5. Report ALL AUC5/10/20,
contrasts, scenes, draws, failure counts, errors above90 degrees, raw/refine
allocation counts and byte-identical output counts. Never select a control as
the new primary after results. A scientific pass requires changed outputs in
both scenes, primary mean delta >=0.5 AUC5 points, positive mean in both scenes,
positive delta in at least8/10 draws, higher mean than confidence and shuffle,
and no increase in failed-pose or severe-error counts versus stock. No trained
model changes after full scores; failure closes this fixed version. A zero-
change output cannot establish a gain: fresh unseeded solver differences on
identical inputs are noise. Report the unchanged-output subset separately.

This is cached-coordinate accuracy research. Measure feature/gate overhead and
training time but claim no saved model inference or end-to-end speedup. Always
raw/stock are recomputed through the exact same qualified compact-final-ba-v1
binary, normalized coordinates, 1000GC iterations and recovery/error rules.
Keep GC and Python OpenCV in separate bounded processes. Prior results cannot
substitute as fresh matched controls for the new full result.

## Reuse, storage and source admission

Use a separately named `plugbench_compact.py` transport. Reuse the existing
prediction, GC, recovery, scoring and supervision functions unchanged; override
only NPZ storage/verification and worker entry dispatch. Store the exact NPZ
byte payloads in phase/chunk ZIP_STORED containers, with member hashes plus
outer-container/index pins. Keep virtual original per-pair names in seals and
receipts. No temporary physical per-pair NPZ forest. Maximum100 pairs/chunk,
64MiB/container,32MiB/member,1GiB aggregate packed+result output; retain all
1500 outcomes/arm. Qualify byte/array-equivalent round trips, tamper/missing/
duplicate/member checks and existing metrics/GT-barrier behavior before smoke.
Preserve the qualified solver, old runner/helpers, all prior output directories
and native caches; no rebuild/install or dependency change is admitted.

New source files are finite: this namespace's analysis.py, execute.py and
audit.py; benchmark compact transport, utility core, training driver, three
small candidate adapters and two fabricated test files. Individual source
<=128KiB, all new source<=256KiB; metadata file<=16MiB, total namespace metadata
<=64MiB. All new campaign files across this namespace, benchmark sources/tests, smoke/full outputs and staging have <=500 physical files
including staging/logs/pins. The complete old global census, all5766 previous
plugbench files and this proposed addition MUST pass the unchanged global
10,000-file/byte guard BEFORE reservation. If it does not fit, stop before
launch and devise smaller lossless storage; do not silently expand old caps.

ROOT publishes full protocol, obtains independent prospective source review,
then disjoint source-only owners implement. ROOT and peers fully review all
saved source/callers before numerical execution. Freeze source, input, runtime,
qualified solver, old evidence, current state and exact plan prefix before each
managed stage; preserve all failed source/history versions. Old AGENTS bytes
are archived before the explicitly user-authorized append of the mandatory
benchmark rule. Historical AGENTS pins resolve only to that exact archive;
the new current file must equal old prefix plus the exact authorized append.
No other historical pin gets an exception. Old scientific outcomes are not
reclassified by that documentation transition.

## Managed stages, cost and failure handling

ROOT alone executes serial stages: saved-CSV analysis<=180s, affected fabricated
offline tests<=180s, train/checkpoint<=600s, eight-pair engineering smoke<=300s,
and full<=7200s. Inclusive campaign allocation8460s within the existing86400s
global total; current charge31605.44731316697s, committed0, families4/6. No
budget reset. Numerical workers<=3600s, four threads,8GiB combined sampled RSS,
fresh ordinary-usage proof and >=8GiB free RAM/32GiB free disk before each stage.
Retain both existing flocks, durable native journals, phase barriers, recorded
process identity/group absence and bounded cleanup. Unknown closure retains
reservation. Use exact Decimal settlement and preserve every command receipt,
including failed terminal commands. No automatic scientific retry or retuning.

Reuse the accepted ROOT observer/accounting definitions by exact source pins,
with a new `joint_utility_v1` branch. Census includes all prior plugbench roots
and at most500 new files, not a second anonymous6000-file allowance. All old
state branches remain byte-value equal; only this branch, authorized budget/
timestamps, owner and append-only plan notes may change. Automation staysPAUSED.
Engineering failure leaves science UNTESTED and adds no failed family. A scored
scientific gate failure increments the new family ONCE (4->5), closes this
version and leaves all historical family accounting unchanged. Success records
development evidence only, never automatic default or final-scene promotion.

After full completion ROOT runs an independently reviewed standard-library
audit of all cells/metrics, packed-member provenance, source/solver pins,
closures, caps and settlement. Record audit cost separately; it makes no native
calls or benchmark rerun. Update the current report/index with exact result and
limits. A research candidate is unfinished until its benchmark and report are
complete, or an explicit incomplete/blocked execution is recorded honestly.
