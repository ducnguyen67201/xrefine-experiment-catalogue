# Verification, ambiguity ranking and spatial coverage

I tested training-free match ranking/selection on the same acquired matches.

**Assignment persistence (AR).** Forbid a candidate and measure the change in the global appearance assignment. Its8-pair/5-draw screen had a favorable mean but failed consistency and cost gates. V9 only recovered scoring of sealed V8 outputs after a join failure; it was not another scientific variant.

**RM046 ambiguity.** Descriptor/scale compatibility against competing matches: pilot baseline50.6181 → candidate46.4751; severe2→6. Shuffle52.8584. The ordering was not simply inverted.

**RM047 coverage.** Descriptor/scale/population alignment was checked away from the pilot; fixed spatial-cell quotas scored49.3525 vs baseline50.3951 and harmed scene0022.

**RM048 quality-gated coverage.** Freeze off-pilot quality cuts and remove mandatory weak-cell picks. Candidate52.3096 vs baseline51.2552, but shuffle53.8842; scene0015 regressed and only6/10 draws improved. Severe3/3/2 for baseline/candidate/shuffle. Promotion gates failed.

**Learning.** Appearance compatibility and geometric coverage do not identify pose-useful matches by themselves. Positive overall point estimates did not survive the declared scene, draw and control requirements.

Core selectors/checks/protocols come from exact executed archives. The reverted active working-tree versions are not substituted for those archives.

## Handpicked files

| File | Purpose |
|---|---|
| [assignment-persistence/runner-v8.py](assignment-persistence/runner-v8.py) | Assignment persistence algorithm |
| [assignment-persistence/reference-v8.py](assignment-persistence/reference-v8.py) | Independent reference |
| [assignment-persistence/evaluate-v9.py](assignment-persistence/evaluate-v9.py) | Evaluator-only recovery of sealed V8 outputs |
| [assignment-persistence/lock.md](assignment-persistence/lock.md) | Frozen scientific lock |
| [ambiguity-rm046/verifier.py](ambiguity-rm046/verifier.py) | Evaluated selection logic |
| [ambiguity-rm046/test_verifier.py](ambiguity-rm046/test_verifier.py) | Checks |
| [ambiguity-rm046/PROTOCOL.md](ambiguity-rm046/PROTOCOL.md) | Frozen protocol |
| [coverage-rm047/verifier.py](coverage-rm047/verifier.py) | Evaluated selection logic |
| [coverage-rm047/test_verifier.py](coverage-rm047/test_verifier.py) | Checks |
| [coverage-rm047/PROTOCOL.md](coverage-rm047/PROTOCOL.md) | Frozen protocol |
| [quality-rm048/verifier.py](quality-rm048/verifier.py) | Evaluated selection logic |
| [quality-rm048/test_verifier.py](quality-rm048/test_verifier.py) | Checks |
| [quality-rm048/PROTOCOL.md](quality-rm048/PROTOCOL.md) | Frozen protocol |

Exact original locations, source bindings and hashes are in [PROVENANCE.json](../../PROVENANCE.json). [Back to catalogue](../../README.md).
