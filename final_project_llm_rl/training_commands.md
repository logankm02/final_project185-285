# Part 1 commands:

## DPO:

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
--algo dpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_dpo_beta01_v1 \
--beta 0.1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 100 \
--eval_interval 100 \
--save_interval 100 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_dpo_beta01_v1
```

## IPO:

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
--algo ipo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_ipo_v1 \
--beta 0.1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 100 \
--eval_interval 100 \
--save_interval 100 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_ipo_v1
```

## AOT:

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
--algo aot \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_aot_beta02_v1 \
--beta 0.2 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 50 \
--eval_interval 50 \
--save_interval 50 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_aot_beta02_v1
```

## Reward Model:

```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_v1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--grad_accum_steps 4 \
--lr 3e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--eval_interval 25 \
--save_interval 50 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_reward_model_v1
```

## GRPO:

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
--algo grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/
step_000445/adapter \
--output_dir /vol/runs/wildchat_min4_judged_5k_grpo_rm445_v1 \
--steps 25 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_grpo_rm445_v1
```

## DrGRPO:

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
--algo dr_grpo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/
step_000445/adapter \
--output_dir /vol/runs/wildchat_min4_judged_5k_drgrpo_rm445_v1 \
--steps 25 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_drgrpo_rm445_v1
```

## GSPO:

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
--algo gspo \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_gen \
--eval_split test_gen \
--reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
--reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/
step_000445/adapter \
--output_dir /vol/runs/wildchat_min4_judged_5k_gspo_rm445_v1 \
--steps 25 \
--batch_size 16 \
--group_size 4 \
--min_new_tokens 32 \
--max_new_tokens 256 \
--temperature 0.8 \
--top_p 0.95 \
--lr 1e-5 \
--grad_accum_steps 2 \
--ppo_epochs 2 \
--minibatch_size 8 \
--clip_eps 0.2 \
--kl_coef 0.01 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--eval_limit 32 \
--eval_interval 25 \
--save_interval 25 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_gspo_rm445_v1
```

# Part 2 Commands:

## AOT-weighted:

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
--algo aot_weighted \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_aot_w \
--beta 0.2 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 50 \
--eval_interval 50 \
--save_interval 50 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_aot_w
```

## APO-Zero:

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
--algo apo_zero \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_apo_zero \
--beta 0.1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 100 \
--eval_interval 100 \
--save_interval 100 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_apo_zero
```

## APO-Down:

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
--algo apo_down \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
--dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
--train_split train_prefs \
--eval_split test_prefs \
--generation_split test_gen \
--output_dir /vol/runs/wildchat_min4_judged_5k_apo_down \
--beta 0.1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--grad_accum_steps 4 \
--lr 5e-5 \
--num_train_epochs 3 \
--max_prompt_tokens 700 \
--max_response_tokens 512 \
--generation_eval_limit 32 \
--generation_eval_max_new_tokens 256 \
--generation_eval_every 100 \
--eval_interval 100 \
--save_interval 100 \
--wandb_enabled \
--wandb_project llm-rl-final-project \
--wandb_name wildchat_min4_judged_5k_apo_down
```

# Online Part 2 Readme:

## Prereqs:

Assume that WildChat dataset is already on Modal volume at `/vol/synthetic_datasets/wildchat_min4_judged_5k_v1`

Upload Dataset to Modal if needed

```bash
uv run modal volume put llm-rl-final-project-volume dataset/wildchat_min4_judged_5k_v1 /synthetic_datasets/
```

All commands are run from the `final_project_llm_rl` folder

## Common variables

```bash
export DATASET="/vol/synthetic_datasets/wildchat_min4_judged_5k_v1"
export RM_ADAPTER="runs/wildchat_reward_model_v2/checkpoints/step_000445/adapter"
export TAG="wildchat_rm445_repro_$(date +%Y%m%d_%H%M%S)"
```

## Run directory names

The training commands below write checkpoints to `--output_dir runs/${TAG}_...`, so your rerun will create a new folder under `/vol/runs/` using your current TAG value

In our final reported runs, the concrete run directories were

- `runs/wildchat_rm445_20260511_231357_online_dpo`
- `runs/wildchat_rm445_20260511_231357_online_aot`
- `runs/wildchat_rm445_20260511_231357_dr_grpo`

If you re-run, list checkpoints using your new run directory name (like `runs/${TAG}_online_dpo`), not the fixed names above

### How to list runs

```bash
uv run modal volume ls llm-rl-final-project-volume /runs | grep -E "online_dpo|online_aot|dr_grpo"
```

### Verify reward-model adapter exists on Modal

```bash
uv run modal volume ls llm-rl-final-project-volume \
  /runs/wildchat_reward_model_v2/checkpoints/step_000445/adapter
```

## Training Online DPO: (Best Online Method)

Runs the replay-based online preference pipeline (`train_rm_online_pref.py`) with `--algo dpo`

### Command:

```bash
uv run modal run --detach scripts/modal_train.py::rm_online_pref_train_remote -- \
  --algo dpo \
  --dataset_name "$DATASET" \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_adapter_path "$RM_ADAPTER" \
  --output_dir "runs/${TAG}_online_dpo" \
  --steps 200 \
  --prompt_batch_size 16 \
  --group_size 4 \
  --updates_per_rollout 4 \
  --per_device_train_batch_size 8 \
  --grad_accum_steps 2 \
  --lr 5e-6 \
  --beta 0.1 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name "${TAG}_online_dpo"
```

After training, checkpoints are saved on the Modal volume under `/vol/runs/wildchat_rm445_20260511_231357_online_dpo/checkpoints/`

### List available checkpoints:

```bash
uv run modal volume ls llm-rl-final-project-volume \
  /runs/wildchat_rm445_20260511_231357_online_dpo/checkpoints
```

If re-running with TAG, list checkpoints for your new run

```bash
uv run modal volume ls llm-rl-final-project-volume "/runs/${TAG}_online_dpo/checkpoints"
```

Run produces checkpoints at:

- `step_000025`
- `step_000050`
- `step_000075`
- `step_000100`
- `step_000125`
- `step_000150`
- `step_000175`
- `step_000200`

Each checkpoint contains a LoRA adapter at `/vol/runs/wildchat_rm445_20260511_231357_online_dpo/checkpoints/<STEP>/adapter`

For our final Part 2 online submission, we used `/vol/runs/wildchat_rm445_20260511_231357_online_dpo/checkpoints/step_000200/adapter`

## Training Online AOT:

Runs the replay-based online preference pipeline (`train_rm_online_pref.py`) with `--algo aot`

### Command:

```bash
uv run modal run --detach scripts/modal_train.py::rm_online_pref_train_remote -- \
  --algo aot \
  --dataset_name "$DATASET" \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_adapter_path "$RM_ADAPTER" \
  --output_dir "runs/${TAG}_online_aot" \
  --steps 200 \
  --prompt_batch_size 16 \
  --group_size 4 \
  --updates_per_rollout 4 \
  --per_device_train_batch_size 8 \
  --grad_accum_steps 2 \
  --lr 5e-6 \
  --beta 0.1 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name "${TAG}_online_aot"
```

After training, checkpoints are saved on the Modal volume under `/vol/runs/wildchat_rm445_20260511_231357_online_aot/checkpoints/`

### List available checkpoints:

```bash
uv run modal volume ls llm-rl-final-project-volume \
  /runs/wildchat_rm445_20260511_231357_online_aot/checkpoints
```

If re-running with TAG, list checkpoints for your new run

```bash
uv run modal volume ls llm-rl-final-project-volume "/runs/${TAG}_online_aot/checkpoints"
```

Run produces checkpoints at:

- `step_000025`
- `step_000050`
- `step_000075`
- `step_000100`
- `step_000125`
- `step_000150`
- `step_000175`
- `step_000200`

Each checkpoint contains a LoRA adapter at `/vol/runs/wildchat_rm445_20260511_231357_online_aot/checkpoints/<STEP>/adapter`

## Training DrGRPO:

Runs the GRPO-family online RLHF trainer (`train_rm_grpo.py`) with `--algo dr_grpo`

### Command:

```bash
uv run modal run --detach scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo dr_grpo \
  --dataset_name "$DATASET" \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_adapter_path "$RM_ADAPTER" \
  --output_dir "runs/${TAG}_dr_grpo" \
  --steps 200 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 5e-6 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.05 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name "${TAG}_dr_grpo"
```

After training, checkpoints are saved on the Modal volume under `/vol/runs/wildchat_rm445_20260511_231357_dr_grpo/checkpoints/`

### List available checkpoints:

```bash
uv run modal volume ls llm-rl-final-project-volume \
  /runs/wildchat_rm445_20260511_231357_dr_grpo/checkpoints
```

If re-running with TAG, list checkpoints for your new run

```bash
uv run modal volume ls llm-rl-final-project-volume "/runs/${TAG}_dr_grpo/checkpoints"
```

Run produces checkpoints at:

- `step_000025`
- `step_000050`
- `step_000075`
- `step_000100`
- `step_000125`
- `step_000150`
- `step_000175`
- `step_000200`

Each checkpoint contains a LoRA adapter at `/vol/runs/wildchat_rm445_20260511_231357_dr_grpo/checkpoints/<STEP>/adapter`

## Build Part 2 online_best.jsonl from a chosen checkpoint:

This generates the final Part 2 online submission JSONL from a specific checkpoint adapter using the fixed public prompts file

### Build on Modal (Ex: Online DPO step_000200)

```bash
uv run modal run --detach scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_rm445_20260511_231357_online_dpo/checkpoints/step_000200/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/part2_online_best.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0 \
  --per_device_eval_batch_size 8
```

### Download locally (for local autograder)(Ex: Online DPO step_000200)

```bash
mkdir -p llm_rl_final_proj_public_submission/part2_candidates

uv run modal volume get llm-rl-final-project-volume \
  /submissions/part2_online_best.jsonl \
  llm_rl_final_proj_public_submission/part2_candidates/online_best_wildchat_rm445_20260511_231357_online_dpo_step_000200.jsonl
```

### Set as active Part 2 online file for local autograder to run:

```bash
mkdir -p llm_rl_final_proj_public_submission/part2
cp llm_rl_final_proj_public_submission/part2_candidates/online_best_wildchat_rm445_20260511_231357_online_dpo_step_000200.jsonl \
   llm_rl_final_proj_public_submission/part2/online_best.jsonl
```
