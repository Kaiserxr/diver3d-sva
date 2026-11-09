# Lab machine runbook

Step-by-step commands for setting up and running experiments on the lab PC.
Run them **in order**, one block at a time, in a terminal on the lab PC.

- 📋 **= paste this output back** before going on. If a command prints an error you
  don't understand, stop and paste it; don't improvise a fix.
- Anything that runs longer than a couple of minutes runs inside **tmux**, so an AnyDesk
  drop doesn't kill it (see [tmux in 60 seconds](#tmux-in-60-seconds)).
- You don't need `sudo` for any step.

## The machine and the versions we use

| | Lab PC | Why it matters |
|---|---|---|
| GPU | RTX 5080, 16 GB | A **Blackwell** GPU (`sm_120`). It only runs code built for **CUDA 12.8 or newer**; older builds fail with *"no kernel image is available"* |
| Driver | 580.178, supports up to CUDA 13.0 | Runs CUDA 12.8 code fine |
| OS | Ubuntu 24.04, gcc 13 | |
| CPU / RAM / disk | 16 cores, 60 GB, 53 GB free | Plenty. Setup uses about 25 GB |
| Missing | conda, nvcc | We install both in your home folder, no admin needed |

Chosen versions:

| Piece | Version | Note |
|---|---|---|
| Python | 3.10 (conda env `sva`) | well tested with OpenPCDet |
| PyTorch | 2.7.1 + CUDA 12.8 | first PyTorch line built for Blackwell |
| CUDA toolkit (nvcc) | 12.8.1, in `~/cuda-12.8` | must match PyTorch's CUDA to build OpenPCDet's extensions |
| spconv | `spconv-cu126` (newest wheel) | **least certain piece on Blackwell.** Step 8 tests it; a source-build fallback is listed |
| numpy | 1.26.4 | OpenPCDet and numba are not numpy-2 ready |
| OpenPCDet | commit `233f849`, pinned by our setup script | |

Folder layout used everywhere below:

```
~/projects/diver3d-sva/      this repo (code)
~/data/diver3d_v2/           the dataset, as downloaded, never edited
~/cuda-12.8/                 CUDA toolkit
~/miniforge3/                conda
```

## tmux in 60 seconds

tmux keeps a terminal running on the lab PC even when AnyDesk disconnects.

| Do this | Command / keys |
|---|---|
| Start a named session | `tmux new -s NAME` |
| **Detach** (leave it running) | press `Ctrl+b`, release, then press `d` |
| List sessions | `tmux ls` |
| **Reattach** after a drop | `tmux attach -t NAME` |
| Scroll back inside tmux | `Ctrl+b` then `[`, arrows / PgUp, `q` to stop |
| End a session | type `exit` inside it |

If `Ctrl+b` doesn't reach tmux through AnyDesk, click into the AnyDesk window first.
Every training run also writes a log file under `~/projects/diver3d-sva/logs/`, so you
can always check progress without attaching: `tail -f ~/projects/diver3d-sva/logs/<file>`.

---

# Part 1: one-time setup

## Step 1: check the compiler

```bash
gcc --version | head -1; g++ --version | head -1; make --version | head -1; unzip -v | head -1
```

📋 Paste the output. If any says "command not found", we need the lab admin to run
`sudo apt install build-essential unzip`. Stop here until that's done.

## Step 2: get the code

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Kaiserxr/diver3d-sva.git
cd diver3d-sva && git log --oneline -1
```

If git asks for a username or password, the repo is still private. Make it public
on GitHub first, or ask me for the private-repo steps.

## Step 3: get the data

Download both files from the competition's Google Drive folder in the lab PC's
browser: `diver3d_v2.zip` and `diver3d_v2.zip.sha256`. They'll land in
`~/Downloads`. Then:

```bash
cd ~/Downloads
cat diver3d_v2.zip.sha256
sha256sum diver3d_v2.zip
```

The two long hex strings must be identical. That proves the 4 GB download isn't
corrupted. Then unpack and check every file:

```bash
mkdir -p ~/data && cd ~/data
unzip -q ~/Downloads/diver3d_v2.zip
ls ~/data
```

The folder that contains `splits.json` is the dataset root. The steps below assume
it's `~/data/diver3d_v2`; if `ls` shows something else, tell me.

```bash
cd ~/data/diver3d_v2 && sha256sum --check --quiet CHECKSUMS.sha256 && echo "DATA OK: all files verified"
```

📋 Paste the two hashes and the last line. It should say `DATA OK`. After that you can
delete the zip to save 4 GB: `rm ~/Downloads/diver3d_v2.zip`

## Step 4: install conda (Miniforge)

```bash
cd ~ && wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
rm Miniforge3-Linux-x86_64.sh
```

**Close the terminal and open a new one** so `conda` is found, then:

```bash
conda --version
```

## Step 5: create the Python environment

```bash
conda create -y -n sva python=3.10
conda activate sva
python --version
```

From now on, **every new terminal or tmux window needs `conda activate sva` first.**
The prompt then starts with `(sva)`.

## Step 6: install the CUDA 12.8 toolkit (about 15 min, use tmux)

This installs only the compiler and libraries into `~/cuda-12.8`. It does **not**
touch the GPU driver.

```bash
tmux new -s setup
```

Inside tmux:

```bash
cd ~ && wget -c https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8 --no-man-page --override --tmpdir=$HOME
~/cuda-12.8/bin/nvcc --version
```

If the download drops, rerun the same `wget -c` line; it resumes. When `nvcc --version`
prints `release 12.8`, delete the 5 GB installer with `rm ~/cuda_12.8.1_*.run`.

Now make the `sva` environment use it automatically:

```bash
conda activate sva
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh <<'EOF'
export CUDA_HOME=$HOME/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0"      # build CUDA code for the RTX 5080 (Blackwell)
EOF
conda deactivate && conda activate sva
which nvcc && nvcc --version | tail -2
```

📋 Paste the last output. It should show `~/cuda-12.8/bin/nvcc` and `release 12.8`.

## Step 7: install PyTorch

```bash
conda activate sva
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python ~/projects/diver3d-sva/scripts/check_gpu_stack.py
```

📋 Paste the summary. `torch + CUDA on GPU` must say **PASS** and mention
`sm_120`. The other lines can still say SKIP.

## Step 8: install spconv and the other packages

```bash
conda activate sva
pip install "numpy==1.26.4" "numba==0.60.0" llvmlite scipy scikit-image tqdm easydict pyyaml \
    tensorboardX tensorboard SharedArray "opencv-python-headless==4.10.0.84" pyquaternion pillow
pip install spconv-cu126
python ~/projects/diver3d-sva/scripts/check_gpu_stack.py
```

📋 Paste the summary. `spconv sparse conv on GPU` must say **PASS**.

> **If spconv FAILS** (typically *"no kernel image"* or *"arch not supported"*), the
> prebuilt wheel lacks Blackwell kernels, and spconv must be built from source for
> `sm_120`. Paste the error first. The fallback, which I'll confirm against your error, is:
>
> ```bash
> pip uninstall -y spconv-cu126 cumm-cu126
> pip install pccm ccimport "pybind11>=2.6.0" fire
> export CUMM_CUDA_ARCH_LIST="12.0"
> git clone --depth 1 https://github.com/FindDefinition/cumm ~/src/cumm && pip install -e ~/src/cumm
> git clone --depth 1 https://github.com/traveller59/spconv ~/src/spconv && pip install -e ~/src/spconv
> ```

## Step 9: set up and build OpenPCDet (about 5–10 min)

```bash
conda activate sva
cd ~/projects/diver3d-sva
bash scripts/setup_openpcdet.sh ~/data/diver3d_v2
cd third_party/OpenPCDet && python setup.py develop 2>&1 | tail -5
python ~/projects/diver3d-sva/scripts/check_gpu_stack.py
```

📋 Paste the setup script's `[1/5]`…`[5/5]` lines and the summary.
`OpenPCDet CUDA ops` must say **PASS**. (`torch_scatter` SKIP is expected; we only need it for DSVT later.)

Record the exact environment so runs can be reproduced:

```bash
pip list 2>/dev/null | grep -iE "^(torch|torchvision|spconv|cumm|numpy|numba|pcdet|scipy|tensorboard) "
```

📋 Paste it. I'll commit it as the environment record.

## Step 10: load data through OpenPCDet

```bash
conda activate sva
python ~/projects/diver3d-sva/scripts/check_dataset.py
```

📋 Paste the output. Expected: `train: split=train, 20861 frames`,
`test: split=val, 2010 frames` (every 4th val frame), point and voxel shapes, some
boxes, and `DATASET CHECK OK` at the end.

---

# Part 2: sanity check, overfit a tiny subset (about 10–20 min)

Before spending hours on real training, prove that the whole pipeline works:
data, targets, loss, evaluation. We train on **105 training frames** (every 200th:
76 contain divers, 100 boxes) and evaluate on those **same** frames. A model that
works should memorise them, so loss falls and scores climb far above normal. If they
don't, something is wired wrong, and a 2-hour run would have been wasted.

Config: `cfgs/diver3d_models/pointpillar_overfit.yaml`, identical to `pointpillar.yaml`
except the lines marked `OVERFIT`.

```bash
tmux new -s overfit
```

Inside tmux:

```bash
conda activate sva
cd ~/projects/diver3d-sva
bash scripts/train.sh pointpillar_overfit sanity --num_epochs_to_eval 5
```

Detach with `Ctrl+b` then `d`. Watch progress from any terminal:

```bash
tail -f ~/projects/diver3d-sva/logs/pointpillar_overfit__sanity__*.log
```

When it finishes (the log ends with `finished ... exit code 0`):

```bash
cd ~/projects/diver3d-sva
grep -E "EPOCH [0-9]+ EVALUATION|  (SDS|mAP3D|AP3D@0.35|AP3D@0.50|APBEV@0.35) " logs/pointpillar_overfit__sanity__*.log
grep "Loss:" logs/pointpillar_overfit__sanity__*.log | sed -n '1p;$p'
```

📋 Paste both outputs.

**What good looks like.** The last loss is much lower than the first, and
APBEV@0.35 on these frames gets close to 1. AP3D stays lower: PointPillars predicts
upright (yaw-only) boxes, and even *perfect* upright boxes only reach AP3D@0.35 ≈ 0.94,
AP3D@0.50 ≈ 0.66 and AOE3D ≈ 65° on Diver3D. That's the "oracle" ceiling, measured on val.

---

# Part 3: baseline, PointPillars, 20 epochs

The real run: all 20,861 training frames, 20 epochs, seed 666, no augmentation (the
paper's protocol for baselines). After training, each epoch's checkpoint is scored on
the val subset (every 4th frame), which is how we pick the best epoch.

Time: unknown until the first epoch finishes. Colab's T4 took about 15 min per epoch;
the RTX 5080 should be several times faster. Expect roughly 1–3 hours in total.

```bash
tmux new -s pp_base
```

Inside tmux:

```bash
conda activate sva
cd ~/projects/diver3d-sva
bash scripts/train.sh pointpillar baseline_s666 --num_epochs_to_eval 20
```

Detach (`Ctrl+b`, `d`). 📋 After about 15 minutes, paste the last 20 lines of the log
so we can estimate the total time:

```bash
tail -n 20 ~/projects/diver3d-sva/logs/pointpillar__baseline_s666__*.log
```

### When it finishes: pick the best epoch, then score it on the full val set

```bash
cd ~/projects/diver3d-sva
grep -E "EPOCH [0-9]+ EVALUATION|  (SDS|mAP3D|AP3D@0.35|AP3D@0.50) " logs/pointpillar__baseline_s666__*.log
```

📋 Paste it. Choose the epoch with the best AP3D@0.35 (the paper's selection rule),
ignoring the first 4 epochs. Then score that one checkpoint on **all** 8,037 val frames.
Replace `N` with the epoch number:

```bash
tmux new -s pp_eval
conda activate sva
cd ~/projects/diver3d-sva/third_party/OpenPCDet/tools
python test.py --cfg_file cfgs/diver3d_models/pointpillar.yaml --extra_tag baseline_s666 \
    --ckpt ../output/diver3d_models/pointpillar/baseline_s666/ckpt/checkpoint_epoch_N.pth \
    --eval_tag full_val_epN --set DATA_CONFIG.SUBSAMPLE.test 1 \
    2>&1 | tee ~/projects/diver3d-sva/logs/pointpillar__baseline_s666__full_val_epN.log
```

📋 Paste the block of scores at the end (SDS … TILT_deg and the TP count).

**How to read it.** The paper's PointPillars row (test split, 2 seeds, trained from
scratch without augmentation) is AP3D@0.35 = 0.190, AP3D@0.50 = 0.018, SDS = 0.370.
Ours is on **val** with **one** seed, so expect the same ballpark, not identical
numbers. Something like 0.1–0.3 AP3D@0.35 is consistent. An order of magnitude off
either way means something is wrong.

---

# Part 4: day-to-day

**Get new code** (after I push changes):

```bash
cd ~/projects/diver3d-sva && git pull
```

If the update notes say so, rerun `bash scripts/setup_openpcdet.sh ~/data/diver3d_v2`.
It's safe to run again.

**Check what's running / reattach:**

```bash
tmux ls
tmux attach -t pp_base
nvidia-smi                      # GPU busy? memory used?
ls -t ~/projects/diver3d-sva/logs | head      # newest logs first
```

**TensorBoard** (loss and score curves). Run it in its own tmux session, then open
<http://localhost:6006> in the lab PC's browser:

```bash
tmux new -s tb
conda activate sva
tensorboard --logdir ~/projects/diver3d-sva/third_party/OpenPCDet/output/diver3d_models --port 6006
```

**Where results live** for run `<tag>` of config `<cfg>`:

| What | Where |
|---|---|
| Full console log, with git commit and exact command | `logs/<cfg>__<tag>__<time>.log` |
| Config copy, OpenPCDet log with fully merged config | `third_party/OpenPCDet/output/diver3d_models/<cfg>/<tag>/` |
| Checkpoints | `.../<tag>/ckpt/checkpoint_epoch_*.pth` |
| Per-epoch scores and prediction files | `.../<tag>/eval/` |

**Reproducibility rules.** Every run gets its own tag (`train.sh` refuses to reuse a
tag that already has checkpoints). Commit config changes before training, so the
commit recorded in the log really is the code that ran. `train.sh` notes uncommitted
changes in the log header.

**Disk:** each run's checkpoints take a few hundred MB. Check with `df -h ~`.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `no kernel image is available for execution on the device` | Something was built without Blackwell (`sm_120`) support | Paste the full error. Which package raised it decides the fix |
| `nvcc: command not found` / `CUDA_HOME` errors while building | `conda activate sva` not run, or Step 6's activate script missing | `conda activate sva`, then `which nvcc` |
| `ModuleNotFoundError: pcdet` | OpenPCDet not built, or wrong env | `conda activate sva`, redo Step 9 |
| `CUDA out of memory` | Batch too big | Add `--batch_size 2` to the `train.sh` command, and note it |
| Training stopped when AnyDesk dropped | It wasn't inside tmux | Always `tmux new -s NAME` first |
| `run '<tag>' already has checkpoints` | Tag reused | Use a new tag, e.g. `baseline_s666_v2` |

## Things to know about the current baseline

- **Yaw-only boxes.** Our `Diver3DDataset` converts each tilted (SO(3)) diver box to the
  tightest upright box, because PointPillars can only predict upright boxes. That's the
  ceiling described in Part 2.
- **Empty frames are skipped in training.** OpenPCDet's base dataset skips any training
  frame without boxes (`pcdet/datasets/dataset.py`, around line 212) and draws another
  frame instead. About 27% of Diver3D frames have no diver, and the paper treats them
  as real negatives. Evaluation still includes them. To revisit later.
- **One seed.** OpenPCDet's `--fix_random_seed` always uses seed 666. A second seed
  needs a small patch, which comes later.
