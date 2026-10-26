import os
import json
import numpy as np
os.environ["NCCL_TIMEOUT"] = "108000"
import sys
import time
import hydra
import wandb
import datetime
import imageio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import default_collate
import ript.utils.utils as utils
from ript.utils.logger import Logger
from ript.utils.dist_utils import sync_rollout_results_via_file
from ript.algos.rl_optimizers import QuestModelAdapter, RolloutGenerator, RLOptimizer
from ript.model_loader import load_quest_model

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(config_path="config", version_base=None)
def main(cfg):
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    # Initialize distributed training
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))
    
    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print('CUDA_VISIBLE_DEVICES:', cuda_visible_devices.split(','))
    device_number = cuda_visible_devices.split(',')[device_id] if cuda_visible_devices else str(device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_number
    print('device_id', device_id)
    print(f'rank {rank} CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Split tasks across GPUs
    all_tasks = cfg.task.task_names_to_use

    if all_tasks is None:
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(cfg.task.benchmark_name.lower())()
        all_tasks = benchmark.get_task_names()
        print('using all tasks from benchmark', benchmark.name)

    rank_to_tasks = {rank_i: [] for rank_i in range(world_size)}
    for task_i, task_name in enumerate(all_tasks):
        rank_to_tasks[task_i % world_size].append(task_name)
    local_tasks = rank_to_tasks[rank]

    # Print from all ranks with synchronization
    print(f'[RANK {rank}] World size: {world_size}, Device: {device}, Tasks: {local_tasks}\n')

    # Create model adapter
    model, _, _, _ = load_quest_model(
        cfg=cfg,
        local_tasks=local_tasks,
        device=device,
        device_id=device_id,
        world_size=world_size,
        use_ddp=True
    )

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    init_state_start = int(os.environ.get('RIPT_INIT_STATE_START', '0'))
    video_count = int(os.environ.get('RIPT_VIDEO_COUNT', '0'))
    video_dir = os.environ.get('RIPT_VIDEO_DIR')
    save_video_fn = None
    if video_dir and video_count > 0:
        os.makedirs(video_dir, exist_ok=True)
        def save_video_fn(video_chw, env_name, episode_index):
            frames = video_chw.transpose(0, 2, 3, 1)
            init_state_index = init_state_start + episode_index
            video_path = os.path.join(video_dir, f'{env_name}_episode_{init_state_index}.mp4')
            imageio.mimsave(video_path, frames, fps=24)
            return video_path

    episode_records = []
    def save_episode_metadata_fn(env_name, episode_index, success, total_reward, episode_steps, video_path):
        episode_records.append({
            'task_name': env_name,
            'episode_index': episode_index,
            'init_state_index': init_state_start + episode_index,
            'success': success,
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'video_path': video_path,
            'checkpoint_path': cfg.checkpoint_path,
            'checkpoint_sha256': os.environ.get('RIPT_CHECKPOINT_SHA256'),
        })

    rollout_dir = os.environ.get('RIPT_ROLLOUT_DIR')
    save_rollout_fn = None
    if rollout_dir:
        os.makedirs(rollout_dir, exist_ok=True)
        def save_rollout_fn(env_name, episode_index, success, total_reward, episode):
            if 'context_tokens' not in episode or 'action_indices' not in episode:
                raise KeyError('Rollout collection requires LiberoRunner_rl context caching')
            init_state_index = init_state_start + episode_index
            np.savez_compressed(
                os.path.join(rollout_dir, f'{env_name}_episode_{init_state_index}.npz'),
                context_tokens=episode['context_tokens'].astype(np.float16),
                action_indices=episode['action_indices'].astype(np.int64),
                policy_inference_steps=episode['policy_inference_steps'],
                success=np.array(success),
                total_reward=np.array(total_reward),
                episode_steps=np.array(len(episode['actions'])),
                init_state_index=np.array(init_state_index),
                task_name=np.array(env_name),
                checkpoint_sha256=np.array(os.environ.get('RIPT_CHECKPOINT_SHA256', '')),
            )

    if rank == 0:
        print('Saving to:', experiment_dir)
        print('Experiment name:', experiment_name)

        wandb.init(
            dir=experiment_dir,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )

        logger = Logger(train_cfg.log_interval)
    else:
        logger = None
    

    # Set up environment runner
    num_parallel_envs = cfg.task.env_runner.num_parallel_envs
    env_runner = instantiate(cfg.task.env_runner, task_names_to_use=local_tasks, num_parallel_envs=num_parallel_envs, reset_type='ori')
    num_candidates = int(os.environ.get('RIPT_NUM_CANDIDATES', '0'))
    policy = model.module
    planning_policy = None
    if num_candidates > 0:
        from planning.planning_policy import CandidatePlanningPolicy
        value_model = None
        value_checkpoint = os.environ.get('RIPT_VALUE_CHECKPOINT')
        if value_checkpoint:
            from planning.outcome_value import OutcomeValueModel
            value_model = OutcomeValueModel.load(value_checkpoint, device)
        planning_policy = CandidatePlanningPolicy(model.module, num_candidates, value_model, device)
        policy = planning_policy
    
    print(f'[RANK {rank}] Conducting rollout evaluation')
    rollout_results = env_runner.run(
        policy,
        n_video=video_count,
        do_tqdm=train_cfg.use_tqdm,
        render=video_count > 0,
        save_video_fn=save_video_fn,
        save_episode_metadata_fn=save_episode_metadata_fn,
        save_rollout_fn=save_rollout_fn,
        init_state_start=init_state_start,
    )
    metadata_path = os.environ.get('RIPT_EPISODE_METADATA_PATH')
    if metadata_path:
        with open(metadata_path, 'w') as handle:
            json.dump({
                'experiment_id': cfg.exp_name,
                'evaluation_seed': cfg.seed,
                'reset_type': cfg.task.env_runner.reset_type,
                'init_state_start': init_state_start,
                'episodes': episode_records,
            }, handle, indent=2)
    if planning_policy is not None:
        planning_metrics_path = os.path.join(os.path.dirname(metadata_path), 'planning_metrics.json')
        with open(planning_metrics_path, 'w') as handle:
            json.dump({'summary': planning_policy.summary(), 'records': planning_policy.records}, handle, indent=2)
    print(f'[RANK {rank}] Rollout results: {rollout_results}')
    sync_rollout_results_via_file(rollout_results, logger, 0)
    
    # Synchronize all processes
    dist.barrier()
    
    # Clean up distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()