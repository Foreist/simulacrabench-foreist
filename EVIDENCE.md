# Local evidence behind the final choice

All numbers are paired per-cell floored log-score differences (nats per cell) on
public proxy surveys built to the instruments' row counts, with 95% intervals from
a respondent-cluster bootstrap (2,000 draws). Decision rules were written down
before each result was computed. These are local measurements, not organizer
results, and several proxy holdouts were reused across screens.

## Candidate choice: RC6 vs RC6 + 20% Kumo relational model

| Proxy (instrument shape) | Kumo mix − RC6 |
|---|---|
| REACH MCNA Iraq (UNHCR) | +0.000408 [+0.000248, +0.000567] |
| PIAAC UK (World Bank) | +0.000227 [+0.000074, +0.000375] |
| Afrobarometer R9 (UNICEF) | +0.000155 [+0.000009, +0.000296] |

Rejected by the pre-registered "no hidden loser" condition: on Afrobarometer the
not-asked cells (26% of cells) were worse, −0.001058 [−0.001172, −0.000953]. The
gain came from hedging rather than sharpening. RC6 was submitted.

## Execution under the Test budget

The exact archive was rehearsed under Test conditions (one import, three calls,
TRAIN+DEV visible, 3,600 s, 16 GB, no network) on an RTX 3090:

| Condition | Total time | TabPFN-3.5 Fast items (final pass) |
|---|---|---|
| GPU shared with unrelated jobs | 3,097 s | 12/12, 31/76, 30/200 |
| Exclusive GPU | 2,658 s | 12/12, 75/76, 188/200 |

TabICLv2, TabPFN-3 and all neural models covered every item on the exclusive GPU.
On the same respondents, the near-complete run beat the starved run by +0.003646
[+0.003056, +0.004232] (UNHCR-shaped), +0.002059 [+0.001387, +0.002654]
(World Bank-shaped) and +0.000559 [+0.000077, +0.001020] (UNICEF-shaped). Deadline
guards never fail loudly; they trade foundation-model coverage for safety.

## Changes rejected by pre-registered checks

| Change | Result |
|---|---|
| TabPFN-3.5 full checkpoint instead of Fast (all 200 MCNA items, fixed components) | −0.0000162 [−0.000155, +0.000122] |
| Five-fold context averaging | all intervals include 0, at 3.6–4.1× the cost |
| Cheaper-first foundation-model deadline order | +0.000129 (exclusive GPU), −0.000127 [−0.000178, −0.000075] (shared GPU) |
| Averaging the not-asked probability across sibling items | +0.00004 (MCNA), 0 elsewhere |
| Answer-conditional temperature on the pooled forecast | selected T = 1.00; oracle ceiling +0.00002 |

The pooled forecast was already calibrated (expected calibration error 0.004 on
answer cells); over-confidence appeared only in the lower-weighted TabICLv2 and
TabPFN-3 components.

## Development leaderboard

From the scorer's published mechanism, the DEV Laplace noise scale was 3.0%,
6.6% and 8.8% of one display bin (UNICEF, World Bank, UNHCR), so DEV displays were
effectively rounded true scores.
