# one_leg: what was tried, what runs now

Target: 56% success over 100 episodes at randomness=low (HDFlow, ICML 2026, Table 1; DP 51,
JUICER 68, DD 60, HDMI 66, SHD 71 on the same row). HDFlow specifies the data (100 success +
50 failure scripted demos, RGB front + wrist + 14-d proprio, 700 steps) but not one Diffuser
hyperparameter, so the encoder, action space and execution scheme are ours.

## Pipeline 1 (2026-08, retired): frozen R3M features in the trajectory

Diffuse `[8-d delta action | 4110-d R3M+proprio]` conditioned on one frame, execute a_0..a_7.
Best 13% at dim 384 after ~900 eval episodes of ablations: value guidance (dead constant, then
repaired, no effect), success-only data, prev-action conditioning, H=64, T=64, absolute actions,
state tracking -- all flat or worse. Measured causes: recorded deltas are feedback terms (open-loop
replay of a demo fails 0/8), and the first-action error (7-9 mm) exceeded the demos' own per-step
motion in the fine phase. Everything from that phase (scripts, analysis, tests) is archived at
`/scratch2/$USER/maze2d-furniturebench/attic_r3m_pipeline_2026-09-06/`.

## Pipeline 2 (2026-09-06, current): image-conditioned Diffuser, absolute targets

Diffuse `[10-d absolute target (pos, rot6d, grip) | 16-d proprio]`, condition on the two frames
through an end-to-end ResNet18 (the Diffusion Policy recipe, which HDFlow's own DP row puts at
51%), execute the next targets open loop against the live pose, replan. Details in the README.

| run | change | success (n=100) |
| --- | --- | --- |
| `img128` s0 | ResNet18, 100 + 50 demos, guided, rp8 | 12% (3090) |
| `img128` s1 | seed 1 | 9% (3090) |
| `img128_sonly` | `--n_failure 0` | 20% (3090) |
| `img128_sonly` | same checkpoint, unguided | **35%** (3090) |
| `img128_sonly` | same checkpoint, guided, `replan_every=4` | 33% (3090) |
| `img128_sonly` | same checkpoint, unguided, `replan_every=4` / `2` | 36% / 38% (3090) |
| `dino128` s0 | frozen DINOv2 ViT-B/14, 100 + 50 demos | 6% (3090) |
| `img128` s0 | re-evaluated on a 4090 | 6% (GPU generation is not a systematic factor) |
| `img128_v2` | success-only + cosine LR to 0 + EMA 0.9999 (the DP schedule), unguided, `replan_every=2` | 19% (3090) |
| `dino128_v2`, `img250_v2` | the same schedule with DINOv2 / with 250 success demos | cancelled at 179k / 39k steps (2026-09-07) |

Failure mode of `img128` s0 (video, 200k steps): tabletop grasp and move succeed, the leg is picked
and carried to the corner, then the insertion misses and the policy retreats to a neutral pose and
idles. Successful episodes take 425-690 steps (demos: 398-592), so speed is not the issue;
alignment precision at insertion is. The failure demos are all 600-step collector timeouts
(the scripted policy stuck after a missed insertion), which is why mixing them in hurts the prior.

What moved the number, in order: dropping the failure demos (12 -> 20), dropping value guidance
(20 -> 35), replanning every 2 targets instead of 8 (35 -> 38, within the n=100 noise of the 4
setting). What did not move it: a frozen DINOv2 encoder (6%), and the Diffusion-Policy schedule
(`img128_v2` reaches a0 loss 0.0014 against 0.0035 at 200k, i.e. a tighter fit of the same 100
demos, for half the success rate).

`OneLegConfig` now defaults to the 38% recipe -- success-only demos, constant lr, EMA 0.995 every
10 steps, `replan_every=2`, `plan_batch_size=1`, guidance opt-in via `--guide` -- so the README's
train and eval commands reproduce it. The failure demos stay on disk and `--n_failure 50`
restores HDFlow's split. Re-evaluating a checkpoint trained before this change needs
`--override plan_batch_size=1 replan_every=2`, since the config travels inside the checkpoint.

Best result: **38%** (`img128_sonly`, unguided, `--override plan_batch_size=1 replan_every=2`),
against the 50% goal and HDFlow's 56. Untested levers: more demos (`data/one_leg_low_250` holds
250 successes, built 2026-09-07), a second training seed at the best recipe, and a fine-tuned
(rather than frozen) DINOv2.
