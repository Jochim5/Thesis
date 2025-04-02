from typing import List, Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def eval_policy(
    model: PPO,
    body: np.ndarray,
    env_name: str,
    n_evals: int = 1,
    n_envs: int = 1,
    connections: Optional[np.ndarray] = None,
    render_mode: Optional[str] = None,
    deterministic_policy: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> List[float]:
    """
    PPO 模型，用于生成动作
    机器人身体的定义（数组形式）
    环境名称（Evogym 等）
    评估次数
    并行运行环境的数量
    机器人连接数据（可选）
    渲染模式（如 human 或 rgb_array，可选）
    是否使用确定性策略（True 固定选择最高概率的动作，False 使用随机采样）。
    随机种子
    是否输出调试信息
    输出：包含 n_evals 个元素的奖励列表

    Evaluate the controller for the robot in the environment. 在给定环境中评估基于 PPO（Proximal Policy Optimization）训练的控制器模型。函数支持：1)使用自定义机器人身体（body）和连接结构（connections）。2)在多个并行环境中运行。3)进行多次评估，并返回所有评估的总奖励。
    Returns the result of `n_evals` evaluations.
    """
    
    def run_evals(n: int) -> List[float]:
        """
        Run `n` evaluations in parallel.运行 n 个并行环境的评估，每个环境的总奖励
        """
        
        # Parallel environments
        vec_env = make_vec_env(env_name, n_envs=n, seed=seed, env_kwargs={
            'body': body,
            'connections': connections,
            "render_mode": render_mode,
        }) # 使用 make_vec_env 创建包含 n 个并行实例的环境， env_kwargs 传递环境初始化的额外参数（如机器人结构和渲染模式）
        
        # Evaluate
        rewards = []
        obs = vec_env.reset() # vec_env.reset(): 重置环境，获得初始观察值
        cum_done = np.array([False]*n) # 累计完成标志，跟踪哪些环境已完成任务
        while not np.all(cum_done): # 检查 cum_done 中是否所有环境都已完成（即所有值为 True）
            action, _states = model.predict(obs, deterministic=deterministic_policy) # model.predict: PPO 模型根据当前观测值 obs 预测下一步的动作。obs: 当前环境的观测值，形状为 (n_envs, observation_dim)。deterministic: 是否使用确定性策略：True: 总是选择概率最大的动作。False: 根据概率分布采样动作（更具探索性）。action: 模型输出的动作，形状为 (n_envs, action_dim)。_states: 当前模型的内部状态（通常在递归神经网络中使用，PPO 默认忽略）。
            obs, reward, done, info = vec_env.step(action) # vec_env.step(action): 对并行环境执行预测的动作。 action: 输入动作数组（每个环境的动作）。obs: 执行动作后新的观测值。reward: 每个环境即时奖励的数组。done: 每个环境是否完成任务的标志数组（布尔值）。info: 额外信息的字典列表（与环境实现相关，可能包含任务完成的具体原因等）。

            
            # Track when environments terminate
            if verbose:
                for i, (d, cd) in enumerate(zip(done, cum_done)):
                    if d and not cd:
                        print(f"Environment {i} terminated after {len(rewards)} steps")
            
            # Keep track of done environments
            cum_done = np.logical_or(cum_done, done) # np.logical_or(x, y): 逐元素计算两个布尔数组 x 和 y 的逻辑 "或" 运算。更新累计完成标志，用于记录哪些环境已经完成任务
            
            # Update rewards -- done environments will not be updated
            reward[cum_done] = 0
            rewards.append(reward)    
        vec_env.close()
        
        # Sum rewards over time
        rewards = np.asarray(rewards)
        return np.sum(rewards, axis=0) # 将奖励按时间步求和，返回每个环境的总奖励
    
    # Run evaluations n_envs at a time
    rewards = [] # 空列表，每次运行 run_evals（一次评估的结果）都会将返回的奖励追加到这个列表中
    for i in range(np.ceil(n_evals/n_envs).astype(int)): # 循环分批执行评估任务，因为可能需要多次运行以完成 n_evals 次评估。n_evals / n_envs: 总次数/可运行环境数=需要的总批次数。np.ceil(n_evals / n_envs):计算评估所需的批次数，向上取整确保所有评估都能完成。.astype(int):将结果转换为整数，因为 range() 需要整数参数。
        rewards.extend(run_evals(min(n_envs, n_evals - i*n_envs))) # 在每一批中，运行 run_evals 并将返回的奖励结果追加到 rewards 列表中min(n_envs, n_evals - i * n_envs):确定当前批次需要运行的环境数量。每批最多运行 n_envs 个环境，但最后一批可能不足 n_envs
    
    return rewards
