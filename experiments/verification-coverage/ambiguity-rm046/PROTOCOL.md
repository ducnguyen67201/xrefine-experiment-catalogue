# RM046 — frozen-feature ambiguity verification, pilot only

Direct authorization: current master user requested testing the training-free ambiguity verifier on the same matches, confidence-only/equal-count/shuffled controls, settings frozen away from pair_0036, and pilot accuracy/severe errors before more training. This is one materially different correspondence-verification mechanism. RM045 attention remains closed/negative; its G0C1 history is not reset. RM046 starts its own G0C0, max2 guard reworks/3 crash repairs AFTER initial across checks/qualification/pilot/audit. Exhausted039/041 administrative forecast/consumer tasks remain closed and are not being retried here.

## Approved scope and design

The user's "Okay, test ..." approves the proposed GT-free verifier/filter placement and controlled pilot, not a full benchmark advance or more training. The preceding [architecture](../.archify/architecture-pair36-match-verification-20261029-183000/pair36-match-verification-architecture.html) and [editable source](../.archify/architecture-pair36-match-verification-20261029-183000/candidate.json) describe that scope. All inference is local CPU. No installations, remote jobs, publication, protected state writes or new research chats/schedulers.

## Frozen settings BEFORE any new MD feature/pose/score work

- Frozen original XFeat weights; Small and all saved coordinates unchanged. No parameter fitting or updates, no new neural head. Capture original XFeat fine logits through a read-only forward hook; replay complete original XFeat* acquisition at top_k10000. Require exact dtype/coordinate/order equality to the original bank before confidence rows bind. Fresh matches never replace the saved population.
- Re-sample frozen dense descriptors at each original acquired endpoint, from that image's actual qualified processed tensor. Use the vendor preprocess_tensor, normalized dense feature map, vendor bicubic sparse interpolator and L2 normalization. Image0 stays at scale1; image1 relative scales are **[0.8,1.0,1.25]**, using bilinear align_corners=False and actual rounded resize dimensions for coordinate mapping. These constants are chosen prospectively, not fitted to MD scores or reference residuals. Original extraction is already dual-scale; this tests a separate fixed-point relative-scale ambiguity check.
- Each scale's score is the minimum of the forward and reverse own-partner cosine minus the best OTHER-partner cosine among the same acquired rows. Candidate score is the minimum across all3 scales. Original out-of-frame rows receive finite score-4; they are retained in audit and are not removed from the baseline. No GT, solver result, scene, pose error or pair36-specific region enters the score.
- Candidate retains **k=min(N,max(5,ceil(N/2)))** rows, stable descending score with original row ID tie-break, then original row order. No threshold search or retained-count sweep. This is a fixed rank policy, not learned confidence calibration.
- Original ScanNet FIT records0,1,2 (room0707) qualify actual feature extraction, original acquisition/confidence alignment and selection before a settings-freeze receipt. No ScanNet GT or DEV/FINAL labels are read; no accuracy-based parameter selection occurs. If qualification fails, repair only engineering/interface errors within counters; do not tune scales/count against scores.

## Eight arms, all declared before scoring

1. star_small: full original acquired bank's frozen Small endpoints (matched B).
2. ambiguity: candidate k-row subset, coordinates exactly selected B bytes.
3. confidence_only: topk by original native fine confidence, same coordinate source.
4. equal_random: uniform fixed-seed k subset of original rows.
5. equal_spatial: deterministic farthest-point k subset in normalized joint4D raw endpoint space, start nearest population center; ties original first row.
6. shuffled_score: candidate's score vector permuted across the same pair's original rows, then same topk/order policy.
7. star_native: full original acquired native endpoints.
8. stock_small: full original sparse/stock Small population, separately named/count.

Random generator is NumPy default_rng(20261030+1009*original_pair_index), one stream per pair: first random subset permutation, then score permutation. Same source IDs in smoke/pilot yield the same frozen subsets; native GC still has fresh unseeded sampling, not common random numbers. All original rows, descriptor vectors,3 scores, original confidence logits, selected IDs and source/input pins are retained. All5 filtered arms have exactly k rows. Other than filtering, no coordinate/solver/recovery/scorer intervention.

## Existing pipeline and qualification

Use the executed045 local process supervisor, qualified original MD image guard and acquired banks. Existing source-qualified gc_worker and recover bodies remain unchanged. Existing paired pose_worker loops remain unchanged; only its two returned hardcoded six-arm COUNT expressions bind to len(plan.methods)=8, with both source hashes retained. This is a minimal arm-count metadata extension, not new pose math or a new general runner. Existing score_worker and scalar metric definitions remain unchanged.

Affected analytic checks → real GT-free ScanNet3 qualification/freeze → original MD engineering slots0..7 x8arms x1 (64GC,128 dependent paired readouts plus64 forced named witnesses) → SAME original039 metadata-selected32 directed pairs (16/scene) x10 x8 (2560GC,5120 dependent paired readouts) → separate saved-data audit. Original K/1024 author processing/RGB guard/isolatedGC threshold1 conf.99999 min=max1000 sampler0/OpenCV/conditional recovery/failure180/ALL rows and pairs persist. Prediction, GC and pose seal globally for each panel before downstream and GT scoring. No full benchmark or additional training in this task, even if pilot passes.

## Frozen pilot decision and reporting

Candidate-B mean AUC5>=0.5 point; positive mean in EACH exposed scene; candidate exceeds EACH confidence/random/spatial/shuffled control's mean AUC5; at least8/10 original global candidate-B draw deltas strictly positive; no GLOBAL or PER-SCENE failure/severe>90 increase vs B; all source/coverage/GT/engineering/audit guards. A positive point score alone cannot pass. Valid negative closes this mechanism without fraction/scale/seed/control promotion or retuning. A pass is only QUALIFIED_FOR_NEXT_STAGE, never a full-main improvement.

Report AUC5/10/20, per-scene/draw results, failures, severe errors, all matched controls and pair36 descriptive outcomes. Descriptive95% fixed-panel solver-repeat intervals use one Python3.11 Random20261027 stream,10000 rows x10 randrange(10), same matrix across all scopes/arms/thresholds. Replicate fsum(drawAUC)/10 then C-minus-control; linear percentile rank9999*p for.025/.975. Intervals are reporting only, not a pass gate, scene-generalization interval or native common-random-number claim. GLOBAL AUC is calculated from all original pairs, not averaged scene AUC.

## Resources, provenance and closure

One stable shared heavy-execution.lock, same dev16777231/inode204740392/mode0600/singlelink, blocking flock through all owned child closure. No editing while holding it. Qualified runtimes hash-authenticated before scientific imports. CPU4/RSS8GiB/free+inactiveRAM8GiB; default disk32GiB plus all oldheld22753572582B and complete7GiB retention envelope. Count retained045+046 outputs plus existing64MiB/128 source/Git reserve; preserve all old files/charges and no compression/deletion credit. Total new output forecast<=1GiB/1024 paths; no weights written. The copied supervisor's3500s enclosing request envelope/240s cleanup remains a conservative bounded run envelope, not claimed available campaign balance or a user worker-hour allocation. Global hard137100 and unmeasured overhead remain; prior cap-minus-covered is only an incomplete upper bound.

Every changed source/test/config has an UNVERIFIED local checkpoint before verification. Preserve each command/raw stdout/stderr/PID birth/group/sample/exit/closure and enclosing cost once; no parent-child/user/sys/tool-wall double charges. Unknown owned closure retains reservation. No new native retry based on scores. Audit uses independently implemented primitive NPZ/scalar pose/AUC code; replaying saved data is distinct from independently repeating extraction/training/native fitting. Report actual audit limitations.
