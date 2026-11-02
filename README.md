# XRefine experiment catalogue

Handpicked research Python and short explanations for Professor review, prepared by Duc Nguyen on 2 November 2026. This is a separate repository from [slam-research](https://github.com/ducnguyen67201/slam-research).

**Accepted full MegaDepth configuration: 55.03757 AUC5 versus matched stock 48.38490.** The current aim is approximately 60 with a defensible new contribution. None of the new candidate mechanisms catalogued here qualified for promotion. Pilot, generated-task, HPatches and ScanNet scores are separate evidence levels.

## Read by technique

| Technique | What was tried |
|---|---|
| [Cross-match context](experiments/cross-match/README.md) | Neighbor means and learned attention, with independent and shuffled controls |
| [Generated interaction](experiments/generated-interaction/README.md) | Moving-patch coupling and bridge-pixel belief propagation |
| [Refinement benefit and endpoints](experiments/refinement-benefit/README.md) | Whole-pair raw/refined switching, fixed-source endpoints, per-match action risk |
| [Multi-view coordination](experiments/multi-view/README.md) | Robust consensus across a keypoint's partner views |
| [Uncertainty and pose updates](experiments/uncertainty-and-pose/README.md) | Heatmap uncertainty, RGB interface, robust and pixel-metric pose updates |
| [Recrop and response consensus](experiments/recrop-consensus/README.md) | Repeated crop-and-refine and nine-response agreement |
| [Verification and coverage](experiments/verification-coverage/README.md) | Assignment persistence, ambiguity ranking, spatial coverage, quality gating |
| [Translation diversity](experiments/translation-diversity/README.md) | Geometric diversity with reliability fallback |
| [Pose consensus](experiments/pose-consensus/README.md) | Medoid of cached pose hypotheses and independent reduction |

[Brief professor report in Google Docs](https://docs.google.com/document/d/1FzwdrFsyY4uG61Q-uixnmNRRNN4EfBDmjhWv06ZZ4js).

## What the files represent

The copied files retain their original bytes. Each technique page explains the hypothesis, construction, controls, outcome and evidence level. `PROVENANCE.json` records source locations, archived versions or freeze evidence, exact sizes and SHA256 hashes. Saved executed archives are used where prototypes were subsequently reverted. Historical comments such as "SOURCE ONLY" are preserved; the technique page describes any later evaluation.

This is a source-reading catalogue, not a self-contained training or benchmark package. Historical entry points retain original imports, data paths and execution-state assumptions. Data, weights, native binaries, run state and the complete original pipeline are not bundled. Use the small method modules to inspect the ideas; original drivers require their qualified environment and a new output namespace. No original experiment was rerun to assemble this repository.

Run `python3 tools/verify_catalogue.py` for copy-integrity and syntax checks only. This command does not import experiment code or verify the scientific results.

## Attribution

These are experimental adaptations and research implementations, not claims of ownership or novelty over XRefine, XFeat, robust estimation, attention, uncertainty propagation or consensus. See [ATTRIBUTION.md](ATTRIBUTION.md). No new blanket open-source license is assigned to this collection.
