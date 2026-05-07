# EquiBot — orientation

Stanford IPRL implementation of **EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning** ([arXiv:2407.01479](https://arxiv.org/abs/2407.01479), [project page](https://equi-bot.github.io)).

Research code: 3 commits in history, no tests, no CI. Two methods (EquiBot + DP baseline) on three PyBullet mobile-manipulation tasks (cloth folding, object covering, box closing). Pipeline is **generate demos → train → evaluate**, all driven by Hydra configs.

**This repo is a subset of what's in the paper.** The paper covers 6 simulation tasks and 6 real-robot tasks; only the 3 mobile-manipulation sim tasks above are open-sourced here. Push T, Robomimic Can/Square, and all real-robot infrastructure (ZED2 capture, Grounded SAM segmentation, HaMeR hand parsing, Kinova mobile-base control) are paper-only. The full paper PDF is checked in at `equibot.pdf` for reference.

## Stack

Python 3.10. PyTorch 2.1 + PyTorch3D, PyBullet 3.2.6, Gym 0.26.2, Hydra/OmegaConf, HuggingFace `diffusers`, `wandb`, `einops`, `trimesh`, `robosuite`. Tested on Ubuntu 20.04 / RTX 4090 / CUDA 11.8. README has install commands; conda env is `lfd`.

## Layout

```
equibot/
├── envs/
│   ├── vec_env.py                  # async vec-env interface
│   ├── subproc_vec_env.py          # multiprocess implementation
│   └── sim_mobile/
│       ├── base_env.py             # BaseEnv: PyBullet + multi-camera + gym API
│       ├── {folding,covering,closing}_env.py  # the three tasks
│       ├── generate_demos.py       # CLI: --task_name {fold,cover,close}
│       ├── assets/                 # URDF + meshes + textures (kinova robot, cloth, boxes)
│       └── utils/                  # bullet_robot, multi_camera, transformations, ...
└── policies/
    ├── train.py                    # Hydra entry: python -m equibot.policies.train
    ├── eval.py                     # Hydra entry: python -m equibot.policies.eval
    ├── vec_eval.py                 # vectorized rollout helper
    ├── configs/
    │   ├── base.yaml               # global defaults (incl. wandb entity/project — fill these in)
    │   └── {fold,cover,close}_mobile_{dp,equibot}.yaml
    ├── agents/
    │   ├── dp_agent.py             # DPAgent — baseline diffusion policy
    │   ├── dp_policy.py            # DPPolicy network wrapper
    │   ├── equibot_agent.py        # EquiBotAgent (extends DPAgent with SIM(3) machinery)
    │   └── equibot_policy.py       # EquiBotPolicy network
    ├── vision/
    │   ├── sim3_encoder.py         # SIM3Vec4Latent — outputs dict {inv, so3, scale, center}
    │   ├── vec_pointnet.py         # vectorized PointNet backbone (kNN graph)
    │   ├── vec_layers.py           # VecLinear / VecLNA / etc — SHARED, ~20K LOC, do not read cold
    │   ├── pointnet_encoder.py     # standard (non-equivariant) PointNet
    │   └── misc.py
    ├── datasets/dataset.py         # BaseDataset — loads per-timestep .npz files
    └── utils/
        ├── norm.py                 # Normalizer (min-max, group-wise)
        ├── misc.py                 # env/dataset factories
        ├── media.py                # video/image saving
        ├── diffusion/              # standard diffusion (ConditionalUnet1D, EMAModel, schedulers)
        └── equivariant_diffusion/  # VecConditionalUnet1D — equivariance-preserving variant
```

The load-bearing files are the four under `agents/`, the three SIM(3) files (`sim3_encoder.py`, `vec_pointnet.py`, `vec_layers.py`), and the two `conditional_unet1d.py` files. Most other files are short utilities or asset glue.

## Entry points

```bash
# 1. generate demos
python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/fold \
    --num_demos 50 --cam_dist 2 --cam_pitches -75 --task_name fold

# 2. train (data path MUST end with /pcs)
python -m equibot.policies.train --config-name fold_mobile_equibot \
    prefix=run_name data.dataset.path=../data/fold/pcs

# 3. eval — see README for the four eval setups (Original / R+Su / R+Sn / R+Sn+P)
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix=eval_name mode=eval \
    training.ckpt="[log_dir]/train/run_name_s1/ckpt01999.pth" env.vectorize=true
```

Substitute `cover` / `close` / `dp` as appropriate. wandb entity/project must be filled in at the bottom of `equibot/policies/configs/base.yaml` before anything will run cleanly.

## Configs

Hydra. `--config-name <task>_mobile_<method>` selects the file in `equibot/policies/configs/`. Override anything via `key=value` on the CLI (e.g. `prefix=...`, `data.dataset.path=...`, `env.args.scale_high=2`). `agent.agent_name` is `"dp"` or `"equibot"` and selects which agent class is instantiated.

## Key architectural conventions

These are not obvious from grepping and will save real time on a re-read.

- **Scalar ↔ vector action duality.** EquiBot keeps two representations of actions and states:
  - *scalar* form: `[B, T, num_eef * dof]` — flat, includes gripper.
  - *vector* form: `[B, T, D, 3]` — only the spatial axes, used inside the equivariant network so SO(3)/SIM(3) operations apply correctly.
  - Conversion helpers live on `EquiBotAgent`: `_convert_action_to_vec`, `_convert_action_to_scalar`, `_convert_state_to_vec` (`equibot/policies/agents/equibot_agent.py`). When you see `[..., 3]` tensors in the encoder/UNet, you're in vector form. The gripper component is *only* in scalar form — re-attach after denoising.
  - Concrete shape per robot (paper §E): proprioception `S_t = (S^(x), S^(d), S^(s))` is **13D** = 3 (EE position) + 6 (EE orientation, two columns of rotation matrix) + 3 (gravity direction) + 1 (gripper open scalar). Action `A_t = (A^(v), A^(d), A^(s))` is **7D** = 3 (EE position velocity) + 3 (EE angular velocity, axis-angle) + 1 (gripper). Push T is the exception (3D state, 3D action).

- **Equivariance is hybrid, not pure SIM(3) by construction.** SO(3) (rotation) equivariance is built into the network architecture via vector neurons (`vec_layers.py`) — every linear layer, conv1d, and FiLM layer in the equivariant path is replaced with a vector-neuron analog. **Translation and scale equivariance are achieved by canonicalization, not architecture**: subtract the encoder's predicted centroid `Θ_c` and divide by predicted scale `Θ_s` before diffusion, then multiply outputs back by `Θ_s`. So `_convert_*` and the centering/scaling pipeline aren't just plumbing — they're how SIM(3) becomes SIM(3) and not just SO(3). Disable them and you keep rotation equivariance but lose translation/scale.

- **Encoder outputs a 4-key dict.** `SIM3Vec4Latent.forward()` (`vision/sim3_encoder.py`) returns `{inv, so3, scale, center}` — paper notation `Θ = (Θ_R, Θ_inv, Θ_c, Θ_s)`:
  - `so3` (Θ_R) — rotation-equivariant feature, shape `[B, C, 3]`. From `VecPointNet` backbone.
  - `inv` (Θ_inv) — invariant feature, computed inside the encoder as the inner product of `so3` with a `VecLinear`-projected dual. Used as conditioning vector input to the UNet.
  - `center` (Θ_c) — **geometric** centroid: `pcl.mean(-1)`. NOT learned. Used to center positions/offsets.
  - `scale` (Θ_s) — **geometric** scale: mean norm of centered points. NOT learned. Used to normalize position/velocity magnitudes.
  - The encoder only implements `mode="so3"`; `se3` is a TODO. So at the network level this is SO(3)-equivariant only — translation/scale equivariance comes purely from the centering/scaling pipeline outside the network.
  - Conditioning structure (paper eqs 2–3): `Z_a = f_fuse([A^(v)/Θ_s, A^(d)], A^(s))` and `Z_c = ([Θ_inv, (S^(x) − Θ_c)/Θ_s, S^(d)], [S^(s), pos_emb(k)])`. Output reconstruction: `Â_t = (Â_inv^(v) · Θ_s, Â_inv^(d), Â_inv^(s))`.

- **Two-tier normalization.** Vanilla DP normalizes obs and actions separately with min/max (`Normalizer` in `utils/norm.py`). EquiBot adds an outer global scale: compute mean point-cloud scale `s_pc` and mean action scale `s_ac` from a subset of training data, divide all 3D-vector inputs by `s_pc / s_ac` at the start of the forward pass, multiply back at the output. Goal: keep diffusion inputs in roughly `[-1, 1]`. Scalar features are normalized vanilla-style.

- **Three horizons** (configured per task):
  - `obs_horizon` (~2) — past observation steps used for conditioning.
  - `pred_horizon` (~16) — future action steps the diffusion model predicts.
  - `ac_horizon` (~8) — how many of those predicted steps are actually executed before re-planning.

- **Action mode** (`ac_mode` in config):
  - `"abs"` — world-frame; center/scale from the point cloud are applied before diffusion.
  - `"rel"` — relative; EquiBot scales action magnitudes by the predicted point-cloud scale.

- **Variable-size point clouds.** Per-step PCs have different point counts, so they're held as lists during obs-history collection (not stacked into a tensor). The dataset loader resamples to `num_points` (default 1024). `act()` defensively skips empty PCs.

- **EMA model.** Both `DPPolicy` and `EquiBotPolicy` keep an EMA of the noise-prediction net via `EMAModel`; inference uses the EMA weights.

- **Two diffusion UNets.** `utils/diffusion/conditional_unet1d.py` is the standard FiLM-conditioned UNet1D used by DP. `utils/equivariant_diffusion/conditional_unet1d.py` is a `VecConditionalUnet1D` that splits vector and scalar conditioning so equivariance survives the denoiser — used by EquiBot. Inside the equivariant variant, conv1d layers treat the 3-vector channel as a batch dim, FiLM `f` and `h` parameter networks are replaced with `VecLinear`, and upsampling is unmodified (already SO(3)-equivariant).

- **Diffusion scheduler & step counts.** Paper uses DDPM with **100 denoising steps** for sim, DDIM with **8 steps** for real-robot inference. The choice is a config knob (`model.noise_scheduler`); per-task YAMLs set the actual default. PointNet++ encoder is 4 layers / hidden dim 128 in all sim tasks except Push T (2 layers).

- **`vec_layers.py` is shared infrastructure.** Don't try to read it linearly. Treat `VecLinear`, `VecLNA`, etc. as opaque equivariant analogs of their standard counterparts unless you're specifically debugging the encoder.

- **NaN landmine.** `equibot_agent.py:210-214` has an explicit NaN check that drops to `pdb` — past numerical-stability issues in the vectorized path. If training hangs at a breakpoint, this is why.

- **Demo file naming.** `<task>_<config>_t{:04d}.npz` per timestep, grouped into episodes by prefix. Variable-length episodes via padding.

- **`model.use_torch_compile`** has separate code paths for compiled vs non-compiled inference; toggle via config, not by editing call sites.

## Gotchas

- No tests exist. Validation is by running training/eval and reading wandb.
- wandb entity/project must be set in `configs/base.yaml` or runs error out — there is no offline default.
- The `data.dataset.path` override must end in `/pcs` (see README); the demo generator writes a `pcs/` subdir inside `--data_out_dir`.
- GPU effectively required — pure-CPU runs are not a tested path.
- Eval is parameterized into four setups in the paper (Original / R+Su / R+Sn / R+Sn+P) via `env.args.*` overrides; the differences live in the README, not in separate config files.

## When making changes

- Adding a new task env: subclass `BaseEnv` in `equibot/envs/sim_mobile/`, add a config under `configs/` matching the `{task}_mobile_{method}.yaml` pattern, wire it in `policies/utils/misc.py` (env factory).
- Touching equivariance: changes have to round-trip through scalar↔vector conversions consistently. The NaN check at `equibot_agent.py:210` exists for a reason — don't disable it without replacing it.
- Touching the encoder: read the paper first; the SIM(3) decomposition (rotation, scale, center, invariant features) isn't reconstructable from code alone.

## Pointers for deeper context

- Paper: https://arxiv.org/abs/2407.01479 — local copy at `equibot.pdf`. Section §E (supplementary) has the concrete dimension breakdowns and architecture details. Section §3 has the proposition that an SO(3)-equivariant denoising chain produces an SO(3)-equivariant output distribution (so per-step equivariance suffices).
- **Vector neurons** (Deng et al. 2021): the SO(3)-equivariant primitive that `vec_layers.py` implements. If you need to understand `VecLinear` / `VecLNA`, this is the reference, not the EquiBot paper.
- **EquivAct** (Yang et al. 2023, arXiv:2310.16050) — prior work from the same lab. The `SIM3Vec4Latent` encoder is *reused* from EquivAct; EquiBot's contribution is wrapping it with a diffusion process. If the encoder is acting weird, check EquivAct rather than this paper.
- **Diffusion Policy** (Chi et al. RSS 2023, arXiv:2303.04137) — the DP baseline and the architectural starting point EquiBot modifies.
- Project page: https://equi-bot.github.io — figures and demo videos.
- README at repo root has the full install + run commands; this file complements it rather than replacing it.
