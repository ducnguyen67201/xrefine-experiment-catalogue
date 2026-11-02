# RM048 — quality-gated coverage, same-match pilot only

Direct human instruction on 30 October 2026: test quality-gated coverage on the same matches, without mandatory weak-cell picks; freeze settings away from the pilot; compare aligned-score-only, confidence and shuffled controls; report AUC5/severe; no training. This is a prospective known filtering/stratification adaptation with novelty unestablished, not retuning the closed RM047 candidate. RM047/046/039041 remain closed and immutable.

## Rule frozen BEFORE calibration scores or new MD selection

Use ONLY the three archived RM047 GT-free ScanNet FIT qualification records0,1,2 in room707, with original native descriptors/scales/logits/confidence already independently bound to the original acquired rows. No GT, pose errors, MD scores, DEV labels or FINAL data calibrate the gate.

Pool all surviving native margin observations from these three pairs. Separately pool their native fine-confidence observations. Convert saved float32 values exactly to Python float64, sort, and take order index floor(0.25*(M-1)) with no interpolation. Freeze these two lower-quartile cuts. Eligibility is finite margin>=margin_cut AND confidence>=confidence_cut, inclusive. Marginal quartiles plus AND do not guarantee a retention rate. No cut/quantile/seed/grid/fraction adjustment after calibration or scoring. This is unlabeled calibration, not learned training.

For original N, target=min(N,max(5,ceil(N/2))). Let E be eligible original IDs. k=min(target,|E|). Apply existing 4→2→1 raw-endpoint joint-cell coarsening and exact integer apportionment to ELIGIBLE endpoints only: one seat per occupied eligible cell, then capacities |cell|-1 and largest integer remainder with cell-ID ties. Rank within quotas by aligned margin, row-ID ties; return original-order IDs. No ineligible cell seat, weak fill, min5 clamp or coordinate edit. If fewer than5 survive, retain the actual count and score the ensuing failure180. No pose is omitted.

The rule assures occupied eligible cells at chosen grid only, not physical correctness, hull/conditioning or useful pose support. Exact raw/Small coordinates, original indices/order/calibration and solver/recovery/scorer remain unchanged.

## Ten frozen arms

0 star_small: full original B Small coordinates.
1 quality_coverage: candidate C, eligible margin ranking with eligible-only quotas.
2 confidence_only: global native confidence top-k over ALL original N.
3 equal_random: global uniform k.
4 quality_within_shuffled: SAME true eligible IDs and quotas, margin values permuted within E.
5 quality_joint_shuffled: SAME shared permutation of original (margin,confidence) tuples; reapply frozen gate and eligible-only quotas. This preserves joint distribution, eligible cardinality and k while breaking row association.
6 aligned_topk: global aligned margin top-k over ALL original N.
7 quality_score_only: same true E, global margin top-k, no coverage. Isolates added coverage.
8 star_native: full original native STAR.
9 stock_small: full original stock Small.

NumPy default_rng(20261030+1009*original_pair_ID): first permutation(|E|) for within-gate scores, second permutation(N) shared joint tuples, third permutation(N) for uniform selection. No repeat-dependent subsets. Candidate/arms2..7 all keep exact same k per pair, including0. If |E|<=target, C and arm7 retain all E and are byte-identical; flag SAME_SELECTION. Fresh native GC draws are unseeded and unpaired. Apparent pose differences with identical selections are not method effects. Controls are never promoted after scoring.

## Pipeline and barriers

Reuse executed047 thin adapter over unchanged executed045 supervisor and qualified GC/paired recovery/scalar scorer. Reuse the exact archived047 native feature evidence; NO neural forward, image read, descriptor resampling, re-acquisition or parameter update. New selection NPZs retain full primitive lineage; independent lineage and selection audit replays it. Existing047 seals authenticate input caches only; NEW ALL-pair prediction/GC/pose seals are mandatory before downstream scoringGT. Do not borrow old fit/pose errors.

Named UNVERIFIED checkpoint before changed verification ->15 affected analytic checks -> independent off-pilot lineage/calibration -> publish exact cuts+freeze BEFORE new MD selection -> original8 MD slots0..7 ×10arms ×1 engineering smoke and forced named recovery witnesses -> SAME fixed03932 original directed UIDs16/scene ×10arms ×10freshdraws -> independent saved audit of all outcomes. No full benchmark/training in this request.

Native calls with <5 rows are skipped by qualified original code; empty E/zero mask still produces failure180 in ALL outcome rows. Journal audit derives expected actual GC calls from the saved per-arm counts>=5 and keeps complete outcome denominators separately. Preserve actual legacy+duplicate native replay calls and conditional-fallback counts distinctly.

## Frozen decision and reporting

C−B global mean AUC5>=0.5; positive mean in BOTH scenes; C exceeds EVERY same-count control2..7; >=8/10 strictly positive global C−B draw deltas; no GLOBAL or per-scene failure/severe>90 increase vsB; at least one pair with genuine C versus gate-only selection difference; all engineering/source/row/provenance/GT/audit guards pass. Intervals are reporting-only. Valid negative closes/discards by explicit owned git revert AFTER dependent closure, retaining all sources/raw. No retune or control promotion.

Report all-arm AUC5/10/20 global and each scene, selected/eligible counts, identical selections, pair0013/0036 severe outcomes, failures and severe counts, actual cost/timing scope. Descriptive95% conditional FIXED-PANEL solver-repeat intervals: one Python3.11 Random20261027 10000×10 matrix rng.randrange(10), same matrix for ALL thresholds/scopes/arms, fsum selected draw mean /10, linear percentile9999*p. Not independent-scene CI, native common RNG, iid generalization or gate. Two scenes are exposed development data; settings away from pilot does not restore untouched evaluation.

Historical accepted MAIN original1500×10/806images/two exposed scenes remains55.03756982394344 AUC5, target70 aspirational, no new matched full advance. Pilot cannot replace that score.

## Ownership, limits and closure

RM048 initialG0C0; max2 guard reworks/max3 crash repairs AFTER initial across all stages/chats/audit; no reset. Same master/51c8 owns source, checkpoint and execution; separate existing score_sign_review authored independent selection audit source and reviews saved reductions. No extra chat/model/scheduler. Old shared register is full; existing SOURCEHOST continuation is sole control log; derived local report is not a second register.

SAME ORIGINAL045-onward7GiB/3072 retained envelope includes ALL045046047048 raw, partials/failures/source archives/audits plus64MiB/128 source-Git reserve. New complete upper3GiB/1400 paths, no refill/deletion/compression credit. Fresh CPU4/RSS8GiB/free-inactiveRAM8GiB/default DISK32GiB +old-held22753572582 +whole family/copy forecast. Stable0600/singlelink flock dev16777231/inode204740392 through actual PID-generation/group/FD closure; never unlink/recreate/force-unlock. One heavy job, source work parallel. Same1200s enclosing request guard and240s cleanup; forecast480s from actual047223.76s with cached features, not worker-hour authority or old5215s extrapolation. Original hard137100 preserved; known covered130347.92229020586508947 before new diagram/experiment and unmeasured overhead remains incomplete, not settled balance.

Archive actual executed source, retain original command/stdout/stderr/terminal/PIDbirth/groups/FD/cost for every stage. Enclosing main cost charged ONCE, nested/user/sys/tool clocks not additive. Named sources/tests/config/protocol only in local Git; no reports/raw/weights/vendor/blanket staging/push/reset/clean/install/remote/protected writes. No professor contact.
