import os
from typing import List, Optional
import numpy as np
from ppo.eval import eval_policy
from stable_baselines3.common.callbacks import BaseCallback
# #在训练过程中定期评估策略。自动保存表现最好的模型。
class EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``. 定义了一个名为 EvalCallback 的类，继承自 BaseCallback。作用：在训练期间定期评估策略，并保存性能最好的模型。

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
        self,
        body: np.ndarray, # 机器人的形状定义（numpy 数组）
        env_name: str, # 环境名称
        eval_every: int, # 每隔多少步评估一次策略
        n_evals: int, # 每次评估的次数
        n_envs: int, # 并行环境的数量
        model_save_dir: str, # 模型保存的目录
        model_save_name: str, # 保存的模型名称
        connections: Optional[np.ndarray] = None, # （可选）机器人的连接信息
        verbose: int = 0 # 决定输出的详细程度
    ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        
        self.body = body
        self.connections = connections
        self.env_name = env_name
        self.eval_every = eval_every
        self.n_evals = n_evals
        self.n_envs = n_envs
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir) # 如果模型保存目录不存在，则创建
            
        self.best_reward = -float('inf')
        
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.在训练开始时调用。目前为空，可根据需求初始化逻辑
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction 每次策略开始与环境交互前调用。目前为空。
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.每次调用 env.step() 后触发。self.num_timesteps: 当前训练步数。如果当前步数是 eval_every 的倍数，则调用 _validate_and_save() 方法进行评估和模型保存。返回 True 表示继续训练，返回 False 则提前终止。

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        
        if self.num_timesteps % self.eval_every == 0:
            self._validate_and_save()
        return True
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.每次策略与环境交互结束后调用
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.在训练结束时调用，确保最后保存一次模型
        """
        self._validate_and_save()
    
    def _validate_and_save(self) -> None:
        rewards = eval_policy(
            model=self.model,
            body=self.body,
            connections=self.connections,
            env_name=self.env_name,
            n_evals=self.n_evals,
            n_envs=self.n_envs,
        ) # 调用 eval_policy 函数评估策略，并返回奖励列表
        out = f"[{self.model_save_name}] Mean: {np.mean(rewards):.3}, Std: {np.std(rewards):.3}, Min: {np.min(rewards):.3}, Max: {np.max(rewards):.3}" #计算奖励的均值、标准差、最小值和最大值，并格式化为字符串
        mean_reward = np.mean(rewards).item()
        if mean_reward > self.best_reward:
            out += f" NEW BEST ({mean_reward:.3} > {self.best_reward:.3})"
            self.best_reward = mean_reward
            self.model.save(os.path.join(self.model_save_dir, self.model_save_name)) #如果当前均值奖励 mean_reward 超过之前记录的最佳奖励 self.best_reward：更新 self.best_reward。保存模型到指定目录。
        if self.verbose > 0:
            print(out)
        