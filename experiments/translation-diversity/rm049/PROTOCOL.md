# RM049: translation-normal diversity of frozen Small match rows

Status: prospectively frozen initial source protocol, 31 October 2026. Known adaptation; usefulness and novelty are unvalidated. This protocol precedes changed-code checks, offpilot calibration values and all RM049 numerical execution. RM048/047/046 and other closed questions remain closed. No training, new weights, extraction or image forward. Scope is the existing proposed shared-prepass/selection/fallback architecture in `.archify/architecture-pair38-translation-fix-20261030/translation-aware-selection.html` and its editable `candidate.json`; no hosting or deployment change.

## Populations and fixed controls

Reuse the exact original acquired RAW endpoints p, frozen matched Small endpoints q, processed K/hw, original row IDs/order, native confidence, fine descriptors and margin from the closed authentic 048/047 caches. Authenticate original lineage independently; do not use old seals as admission for a new prediction/GC/pose/scoring phase. Preserve original coordinate bytes and ALL original rows in provenance.

Reuse the 048 true eligible population E: margin >= 0.02997952699661255 AND native confidence >= 0.2813373804092407, inclusive. Independently replay the original pooled FIT707 records0/1/2 lower-quartile rule floor(.25*(M-1)), without changing these cuts. Target min(N,max(5,ceil(N/2))); k=min(target,|E|). Quotas are precisely the 048 4->2->1 joint-cell policy over E using ORIGINAL RAW p0,p1, not Small q. Clamp cell labels only, never endpoints. Every occupied cell gets one slot, then integer largest-remainder capacity apportionment, cell-ID ties. No new eligible gate or weak-cell fill.

Exactly ten arms, in this order:
0 `star_small`: full frozen Small B.
1 `translation_diversity`: normal-diversity C under E quotas.
2 `eligible_confidence`: native confidence rank within the same E quotas.
3 `eligible_margin`: original margin rank within the same E quotas.
4 `eligible_random`: fixed random priorities within the same E quotas.
5 `geometry_shuffled`: full weighted normal contributions permuted WITHIN each fixed E cell, then the exact same normal greedy rule.
6 `global_confidence`: descending native confidence over ALL original N, matched k.
7 `global_aligned`: descending original margin over ALL N, matched k.
8 `star_native`: full original native STAR.
9 `stock_small`: full original stock Small.

Rank ties use original row ID. Return original row order for fitting; save chosen greedy order separately. Quota arms2--4 use their stated ranking without a forced anchor. C and geometry-shuffle begin with the highest native confidence feasible E row, original-ID tie. This explicit anchor difference is part of the declared algorithms. RNG is NumPy default_rng(20261031+1009*original pair_index): first permutation(|E|) supplies random priorities; then, in ascending occupied cell ID, permutation(cell size) permutes each entire (weighted3x3 contribution,weight) tuple. No per-draw selection. All controls share the identical prepasses, reliability and fallback, including smoke/pilot overlaps; charge shared physical work once and include it in each method's logical end-to-end cost.

## Three shared prepasses, before selection and GT

Exactly three fixed populations per unique pair: ALL N Small q rows, all EVEN original row IDs, all ODD original row IDs. Each gets one native GC1000 fit and the existing qualified recovery policies, with original legacy replay and conditional replay retained. These run in isolated qualified native/pose processes, never the prediction/selection process. ALL N recovered conditional-policy R is ALWAYS R0. No best/medoid alternative, extra native fit, auxiliary model promotion, final candidate pose feedback or GT operand.

For each successfully recovered pose form E=[unit(t)]x R from that same recovery; retain the original native E stack/mask and recovery provenance. This algebraic reconstruction is not another fit. Even-fit model scores ONLY odd rows for its declared FIT-EXCLUDED support, and odd-fit scores ONLY even rows. Full-fit support is resubstitution and is not presented as held out. Absolute pixel Sampson residual r=abs(x1^T F x0)/sqrt((Fx0)_x^2+(Fx0)_y^2+(F^T x1)_x^2+(F^T x1)_y^2), F=K1^-T E K0^-1. Fixed support cut is <=1 processed pixel. Weight each row ONLY with its opposite-parity model's residual; E0 residual is never a weight. These are complementary fit checks on one dependent pair, not independent validation. Coherent incorrect consensus can pass every check.

All three success flags, pairwise maximum R angle, maximum unsigned t angle and both opposite-fold support fractions are saved. Finite zero R/t placeholders are admitted only for declared failed recoveries; their spreads are None. SO3/nonzero t required on successful recoveries. Explicit parity fit indices and native inputs must bind actual fits, not only a membership assertion.

## Frozen normal and greedy math

Float64 unit bearings f_s=unit(K_s^-1[q_s,1]); raw n_i=(R0 f0_i) cross f1_i. Preserve raw norms, including tiny normals. Only cap large magnitudes: ntilde=n*min(1,cap/max(norm(n),eps)). cap is the pooled eligible-normal q90 order statistic floor(.90*(M-1)) over ALL three FIT707 calibration records0/1/2, with no omitted record or later parameter selection.

w_i=clip(native_confidence_i,0,1)/(1+(r_i/1px)^2), implemented stably for large finite residuals. A_i=w_i ntilde_i ntilde_i^T. M=sum A_i is symmetrized. Ascending eigenvalues lambda1<=lambda2<=lambda3; absolute PSD tolerance1e-12: reject smaller than -1e-12, otherwise clip negative eigenvalues to0 and retain raw values. eps1e-12, proper-R tolerance1e-5. Q=(lambda2-lambda1)/(trace+eps). Coplanarity statistic lambda1/(trace+eps). Absolute information is (lambda2-lambda1)/row_count, not merely trace. No eigenvector is used to choose the final pose.

Starting from the declared confidence anchor, greedily choose the feasible row maximizing the resulting Q, compare Python round(Q,12), then native confidence, then original row ID. Float64 bounded vectorized eigvalsh batches512. Rank-one first-gap ties use the explicit anchor, not eigensolver noise. Select EXACTLY k under EXACT quotas, including negative improvements. Save only selected order, chosen-step Q/information history and bounded primitive per-row matrices; never all O(E*k) candidate matrices. Global aligned uses its predeclared all-N score ranking; no control is promoted after results.

## Offpilot calibration and common candidate-independent fallback

The ONLY calibration values are the already archived GT-free ScanNet FIT707 records0/1/2 used by048. All three fresh prepasses must qualify. After fixed pooled q90 cap, compute their WHOLE TRUE ELIGIBLE capped/weighted matrices. Bounds are exactly: MAX of the three maximum R spreads; MAX of unsigned t spreads; MIN of their minimum opposite-fold support fractions; MIN of absolute eigengap/eligible row; MAX coplanarity. Strictly positive support, cap and absolute information >eps are required for this premise. If any record fails, preserve a scientific premise failure and UNTESTED final poses; do not omit it, change quantile, loosen tolerances, select another provisional model or rerun a favorable seed.

On MD, evaluate that single shared gate on the WHOLE TRUE E matrix BEFORE any selection, never a selected-set matrix. Require all three successful fits, both spreads <=bounds, heldout support>=bound, absolute info>=bound and coplanarity<=bound, inclusive. Failure/unreliability gives unmodified full B coordinates and N rows identically to arms1--7, with explicit N-vs-k count exception/frequency. Admitted k<5 remains exactly k, with the original native skip/failure180; no fill or pair drop. A common gate rejection is the separately declared full-B fallback exception. Each final selected matrix's information is diagnostic only; no selected-set or per-control gate.

If C==B on ALL pilot pairs or C==eligible_margin on ALL pilot pairs, record SAME_NO_METHOD_EFFECT. Native fit randomness cannot be described as an algorithm effect, and no full run follows.

## Execution, audit and decision

Named source/protocol/config/whole-forecast only are staged into a local UNVERIFIED checkpoint BEFORE affected checks. Preserve every failed source/raw/clock/closure. New049 G0C0, max2 guard and3 crash failures AFTER initial across every phase/auditor/chat; no resets, exemptions or automatic failed-stage replay. No changes to canonical ledger/register/protected history. No new agent/chat/model/scheduler, training, install, blanket stage, push/reset/clean. Independent math is separately authored `audit_reference.py`; producer must not import it as its implementation.

Path: meaningful analytic and production-boundary checks -> fresh GT-free 707 three-prepass calibration/operand audit/rule freeze and measured runtime -> fixed original MD slots0..7 engineering x10arms x1 with forced recovery witnesses -> exact SAME03932 directed pair IDs/16per scene x10fresh GC draws x10arms -> saved independent lineage/normal/selection/scalar audit -> actual positive parent/child/PIDbirth/group/FD/cost closure -> result. All NEW prediction/GC/pose/control/global seals precede GT scoring. Smoke/pilot shared prepasses and subset overlap bind sealed once. Every original pair/failure remains in the denominator; failures180. Corrected1024K, unchanged native GC1000/OpenCV policies and scalar math. Old seals authenticate cache inputs only; no early GT-body or GT-containing NPZ hash workaround.

Pilot gates ALL required: C-B GLOBAL AUC5>=0.5; both scene deltas>0; C beats EVERY filtered control2--7; >=8/10 strictly positive C-B draws; no GLOBAL/per-scene failed or severe>90 increase; genuine selection difference; all integrity/resource/closure checks. Report global AUC of all rows, not mean scene AUC, and AUC5/10/20, failures/severe, all scopes/draws. Descriptive CI ONLY: one Python Random(20261027) 10000x10 solver-draw resampling matrix, linear rank9999 percentiles, math.fsum; no independent-scene inference.

Conditional full1500x10 ONLY if pilot passes and actual full native-feature coverage/binding, fresh capacity and complete measured extraction/native/process/audit/retention forecast all fit. Cached32 is not1500. No sparse/stock substitute or training. Pilot negative: positive closure first, then explicit owned prototype Git revert while preserving raw/executed sources/report. No retune/same-question successor.

Resources: one shared heavy job, stable blocking LOCK_EX on dev16777231/inode204740392/mode0600/singlelink through positive child/group/FD closure, never unlink/forceunlock. CPU4, inclusive sampledRSS8GiB, fresh free+inactive RAM8GiB (NOT old32GiB), defaultDISK32GiB plus OLD_HELD22753572582 plus whole7GiB family/copy/temporary/closure forecast. SAME045..049 lifetime7GiB/3072 incl64MiB/128 sourceGitreserve and all failure archives/raw/audits; no refill/deletion/compression credit. Enclosing request1200s, each child<=3600s and remaining minus240s cleanup. Measured full forecast must fit before any stage. Actual enclosing clocks once, unknown source/review I/O not zero; covered130586.41927758186508947/hard137100 leaves only an incomplete upper6513.58072241813491053 before unknowns, not a spendable promise.

Cleanup history: 31Oct regenerated unused Apple e5 compiled caches were removed. Original weights/runtime binaries/envs/native catalogs/scientific caches retained; reauthenticate actual qualified catalogs and native chain before scientific imports/calls. Cleanup's old32GiB RAM target is historical and is not operative for049.
