# 🚀 ProcTHOR RL 🚀

RL training scripts for learning an agent using ProcTHOR.

## 💻 Installation 💻

### 🐳 Use Docker Image 🐳
Please refer to [docker/README.md](docker/README.md) for more details.

### 🛠️ Manual Installation 🛠️
```bash
export MY_ENV_NAME=procthor-rl
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"

conda create --name $MY_ENV_NAME python=3.9
conda activate $MY_ENV_NAME

pip install -r requirements.txt
pip install --no-cache-dir --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+ca10d107fb46cb051dba99af484181fda9947a28
pip install --no-cache-dir torch==2.0.1 torchvision open_clip_torch objaverse objathor
```

## 🔍 Test prior package 🔍
Please test the package before running the training script:
```bash
python scripts/test_prior.py
```
You should see the loading of the `procthor-10k` dataset successfully.
Otherwise, please make sure have your `.git-credentials` file in the root directory.

## 🏃‍♂️ Few Running Examples 🏃‍♂️

### Training
Please note that we should now use `CloudRendering` for AI2THOR.
If you find errors related to vulkan, please make sure install `vulkan-tools`, `libvulkan1`, and `vulkan-utils` correctly.️

Training without wandb logging:
```bash
python procthor_objectnav/main.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_ddppo \
    agent=locobot \
    target_object_types=robothor_habitat2022 \
    wandb.project=procthor-training \
    machine.num_train_processes=96 \
    machine.num_val_processes=4 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    procthor.p_randomize_materials=0.8 \
    seed=100
```

Training with wandb logging:
```bash
export WANDB_API_KEY=YOUR_WANDB_API_KEY
python procthor_objectnav/main.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_ddppo \
    agent=locobot \
    target_object_types=robothor_habitat2022 \
    wandb.project=procthor-training \
    machine.num_train_processes=96 \
    machine.num_val_processes=4 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    procthor.p_randomize_materials=0.8 \
    wandb.name=YOUR_WANDB_RUN_NAME \
    wandb.project=YOUR_WANDB_PROJECT_NAME \
    wandb.entity=YOUR_WANDB_ENTITY_NAME \
    callbacks=wandb_logging_callback \
    seed=100
```

Training with Codebook bottleneck:
```bash
python procthor_objectnav/main.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_ddppo \
    agent=locobot \
    target_object_types=robothor_habitat2022 \
    wandb.project=procthor-training \
    machine.num_train_processes=96 \
    machine.num_val_processes=4 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    procthor.p_randomize_materials=0.8 \
    seed=100
```

Training with DINOv2 visual encoder:
```bash
python procthor_objectnav/main.py \
    experiment=procthor_objectnav/experiments/rgb_dinov2gru_ddppo \
    agent=locobot \
    target_object_types=robothor_habitat2022 \
    wandb.project=procthor-training \
    machine.num_train_processes=96 \
    machine.num_val_processes=4 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    procthor.p_randomize_materials=0.8 \
    seed=100
```

### 💾 Download Pretrained Checkpoint 💾

Use scripts/download_ckpt.py to download the pretrained checkpoint:
```bash
python scripts/download_ckpt.py --save_dir YOUR_CKPT_DIR --ckpt_ids CKPT_ID
```
Options for `CKPT_ID`: `CLIP-GRU`, `DINOv2-GRU`, `CLIP-CodeBook-GRU`, `DINOv2-CodeBook-GRU`.

### 📊 Evaluation 📊
Evaluate in `ArchitecTHOR`, `ProcTHOR-10k`, `iTHOR`, or `RoboTHOR`:
```bash
export WANDB_API_KEY=YOUR_WANDB_API_KEY
python procthor_objectnav/main.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_ddppo \
    agent=locobot \
    target_object_types=robothor_habitat2022 \
    machine.num_train_processes=1 \
    machine.num_test_processes=20 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    callbacks=wandb_logging_callback \
    seed=100 \
    eval=true \
    evaluation.tasks=["architecthor"] \
    evaluation.minival=false \
    checkpoint=YOUR_CHECKPOINT \
    wandb.name=YOUR_WANDB_RUN_NAME \
    wandb.project=YOUR_WANDB_PROJECT_NAME \
    wandb.entity=YOUR_WANDB_ENTITY_NAME \
    visualize=true \ # store qualitative results
    output_dir=YOUR_OUTPUT_DIR # dir to store both qualitative and quantitative results
```
`evaluation.tasks` can be `architecthor`, `procthor-10k`, `ithor`, or `robothor`.
