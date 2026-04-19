# Core LwF Ablations (120 epochs)

This set is a compact, high-signal ablation suite.

## Config groups

- Baselines:
  - `train_baseline_lwf_off_e120.yaml` (true original behavior)
  - `train_baseline_lambda0_e120.yaml` (LwF pipeline active, zero weight)
- Fixed lambdas across reset intervals:
  - `lambda_lwf in {0.25, 0.5, 1.0, 1.5}`
  - `n_reset in {1, 10, 50}`
- Fluid lambdas:
  - `train_fluid_l0_to1_n10_e120.yaml`
  - `train_fluid_l1_to0_n10_e120.yaml`

Total: 16 configs.

## Suggested seed protocol

Use at least 3 seeds per config, e.g. 4234, 5234, 6234.

Example run:

```bash
python train.py hparams/ablations_core/train_fixed_l1_n10_e120.yaml --data_folder=/path/to/voicebank seed=5234
``
