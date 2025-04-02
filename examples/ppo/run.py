import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ppo.eval import eval_policy
from ppo.callback import EvalCallback
# 使用 PPO 训练强化学习模型，同时定期评估性能并保存最佳模型。
def run_ppo(
    args: argparse.Namespace, # args: 包含从命令行解析得到的参数（例如学习率、总训练步数等）。
    body: np.ndarray, # body: 机器人的形状（numpy 数组）。
    env_name: str,
    model_save_dir: str, # 模型保存目录。
    model_save_name: str, # 模型保存文件名。
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float: #在评估过程中达到的最佳奖励
    """
    Run ppo and return the best reward achieved during evaluation.
    """
    
    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    }) #使用 make_vec_env 创建并行环境。参数：env_name: 环境名称。n_envs=1: 并行环境数量（这里是 1 个）。seed: 随机种子。env_kwargs: 给环境的额外参数，包括 body 和 connections。
    
    # Eval Callback
    callback = EvalCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        verbose=args.verbose_ppo,
    ) #创建 EvalCallback 回调实例，用于评估模型性能并保存最优模型

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=args.verbose_ppo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps, # 每次更新的步数。
        batch_size=args.batch_size, # 小批量数据的大小。
        n_epochs=args.n_epochs, # 每次更新的优化次数。
        gamma=args.gamma, # 折扣因子。
        gae_lambda=args.gae_lambda, # 广义优势估计的 λ 参数。
        vf_coef=args.vf_coef, # 值函数的损失权重。
        max_grad_norm=args.max_grad_norm, # 梯度裁剪阈值。
        ent_coef=args.ent_coef, # 熵项系数（鼓励探索）。
        clip_range=args.clip_range # PPO 的裁剪范围。
    )
    model.learn(
        total_timesteps=args.total_timesteps, # 总训练步数。
        callback=callback, # 见上
        log_interval=args.log_interval # 日志打印间隔（单位：时间步）。
    )
    
    return callback.best_reward # 训练完成后，返回在评估中获得的最佳奖励（由 EvalCallback 记录）。