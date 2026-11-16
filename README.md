# SimulacraBench (NeurIPS 2026) — team foreist

Code for team **foreist**'s Test submission to the
[SimulacraBench](https://simulacrabench.org/) competition
([Codabench 17822](https://www.codabench.org/competitions/17822/)).
Author: Taewoong Kim (independent, South Korea).

| | |
|---|---|
| Test submission | 1054916 (Finished) |
| Test score (exact) | grand skill 0.4867794729 — UNICEF 0.3426596533, World Bank 0.2366865748, UNHCR 0.8809921907 |
| Submitted archive | `rc6-profile.zip`, 657,391,852 bytes, SHA-256 `2c6ea85fa4960e171116989115a67202e0cdc15fa7733969e3ad7d0640f30def` |
| `submission/main.py` | SHA-256 `9eb86ae493d139b74e30a0548dde7f48d6b7de8138d4235eca9d36d0d32460b7` |

## What it does

`submission/main.py` implements the competition's `predict(frame, schema)`. For every
requested item it estimates the answer distribution conditional on the respondent's
GIVEN fields, learned at run time from the visible rows only:

- Dirichlet backoff chains over GIVEN columns (depth up to 12, levels kept only when
  their leave-one-out gain clears two standard errors), split asked/answer models for
  gated items, a full-GIVEN-profile pooling component, the organisers' estimator without
  its language model, and small multi-task neural networks — stacked by EM on
  out-of-fold, floor-aware log scores;
- three fixed public tabular foundation models, TabPFN-3.5 Fast (4 estimators),
  TabICLv2 (1) and TabPFN-3 (2), mixed with the stack using weights chosen on an
  80/20 split of the visible rows, item weights shrunk toward an instrument curve;
- an approximate inverse of the scorer's probability floor before reporting.

The method description (4 pages) is in `report/`.

## Reproducing the submitted archive

The pretrained checkpoints are **not** redistributed here. Obtain them from their
publishers at the exact revisions below (the TabPFN models require accepting the
Prior Labs licence on Hugging Face), together with each repository's licence file
saved under the name shown:

| File | Source @ revision | SHA-256 | Licence (save as) |
|---|---|---|---|
| `tabpfn-v3.5-fast-20260909.safetensors` | `Prior-Labs/tabpfn_3_5` @ `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e` | `14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550` | TabPFN-3.5 Non-Commercial (`LICENSE.tabpfn-3.5`) |
| `tabpfn-v3-classifier-v3_default.ckpt` | `Prior-Labs/tabpfn_3` @ `24a16a89d245878b846555110985634aa2e656d7` | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` | TabPFN-3 Non-Commercial (`LICENSE.tabpfn-3`) |
| `tabicl-classifier-v2-20260212.ckpt` | `jingang/TabICL` @ `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` | BSD-3-Clause (`LICENSE.tabicl`) |

Copies of the three licence texts as they were bundled are in `submission/` for
reference. Then:

```bash
python tools/assemble_zip.py --weights /path/to/checkpoints --out rc6-profile.zip
```

The script verifies every member against `MANIFEST.json` and checks that the rebuilt
archive is byte-identical to the submitted one (same SHA-256). `requirements.txt`
pins `tabpfn==9.2.0` and `tabicl==2.2.0`; the host image supplies torch, numpy,
pandas and scikit-learn. The code requires a CUDA GPU and the bundled checkpoints:
without them it fails at import by design.

## Evidence

`EVIDENCE.md` summarises the local validation behind the final choice, including
the pre-registered checks that rejected six further changes.

## Data

No competition microdata is included or was used outside the official `predict()`
call. Public survey datasets (GSS, Afrobarometer Round 9, OECD PIAAC Cycle 1 UK
PUF, REACH Iraq MCNA VII/VIII, Open Psychometrics RIASEC) were used offline only, to
build proxy tasks for validation; they are not redistributed here.

## Licence

The code written for this submission (`submission/main.py`, `tools/`) is released
under the MIT licence (`LICENSE`). The pretrained checkpoints and their licence
files remain under their publishers' terms; the TabPFN checkpoints are licensed for
non-commercial use only.
