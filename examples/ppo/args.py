import argparse

def add_ppo_args(parser: argparse.ArgumentParser) -> None:
    """
    Add PPO arguments to the parser 为解析器添加 PPO 参数
    """
    
    ppo_parser: argparse.ArgumentParser = parser.add_argument_group('ppo arguments')
    
    ppo_parser.add_argument(
        '--verbose-ppo', default=1, type=int, help='Verbosity level for PPO: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages (default: 1)' # PPO 的信息级别：0 表示无输出，1 表示信息（如使用的设备或包装器），2 表示调试信息（默认：1）
    )
    ppo_parser.add_argument(
        '--learning-rate', default=2.5e-4, type=float, help='Learning rate for PPO (default: 2.5e-4)' # PPO 的学习率，控制模型参数更新的步长。默认值: 2.5×10^−4
    )
    ppo_parser.add_argument(
        '--n-steps', default=128, type=int, help='The number of steps to run for each environment per update for PPO (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) (default: 128)' # 每次更新中，每个环境运行的步数（即 rollout 缓冲区大小 = n_steps * n_envs）。默认值: 128
    )
    ppo_parser.add_argument(
        '--batch-size', default=4, type=int, help='Mini-batch size for PPO (default: 4)' # PPO 的小批量训练数据的大小。默认值: 4
    )
    ppo_parser.add_argument(
        '--n-epochs', default=4, type=int, help='Number of epochs when optimizing the surrogate objective for PPO (default: 4)' # 每次更新中，优化目标的迭代次数。默认值: 4
    )
    ppo_parser.add_argument(
        '--gamma', default=0.99, type=float, help='Discount factor for PPO (default: 0.99)' # 折扣因子，控制未来奖励对当前决策的影响。默认值: 0.99（即未来奖励逐渐衰减）。
    )
    ppo_parser.add_argument(
        '--gae-lambda', default=0.95, type=float, help='Lambda parameter for Generalized Advantage Estimation for PPO (default: 0.95)' # GAE（广义优势估计）的 Lambda 参数，平衡偏差和方差以稳定策略优化。默认值: 0.95
    )
    ppo_parser.add_argument(
        '--vf-coef', default=0.5, type=float, help='Value function coefficient for PPO loss calculation (default: 0.5)' # 值函数损失在 PPO 总损失中的权重，用于调整值函数和策略优化的平衡。默认值: 0.5
    )
    ppo_parser.add_argument(
        '--max-grad-norm', default=0.5, type=float, help='The maximum value of the gradient clipping for PPO (default: 0.5)' # 梯度裁剪的最大值，用于避免梯度爆炸。默认值: 0.5
    )
    ppo_parser.add_argument(
        '--ent-coef', default=0.01, type=float, help='Entropy coefficient for PPO loss calculation (default: 0.01)' # 熵项系数，鼓励探索的权重。默认值: 0.01
    )
    ppo_parser.add_argument(
        '--clip-range', default=0.1, type=float, help='Clipping parameter for PPO (default: 0.1)' # PPO 中的重要参数，用于裁剪策略更新时的比率变化范围，控制训练的稳定性。默认值: 0.1
    )
    ppo_parser.add_argument(
        '--total-timesteps', default=1e6, type=int, help='Total number of timesteps for PPO (default: 1e6)'  # PPO 的总时间步数，决定训练过程运行的总时间。默认值: 1×10^6 （100 万步）。
    )
    ppo_parser.add_argument(
        '--log-interval', default=50, type=int, help='Episodes before logging PPO (default: 50)' # 每隔多少个 episode 记录日志，用于监控训练过程。默认值: 50
    )
    ppo_parser.add_argument(
        '--n-envs', default=1, type=int, help='Number of parallel environments for PPO (default: 1)' # PPO 训练的并行环境数量，增加环境可以提升训练速度。默认值: 1
    )
    ppo_parser.add_argument(
        '--n-eval-envs', default=1, type=int, help='Number of parallel environments for PPO evaluation (default: 1)' # PPO 评估的并行环境数量。默认值: 1
    )
    ppo_parser.add_argument(
        '--n-evals', default=1, type=int, help='Number of times to run the environment during each eval (default: 1)' # 每次评估中运行环境的次数，用于提高评估的稳定性。默认值: 1
    )
    ppo_parser.add_argument(
        '--eval-interval', default=1e5, type=int, help='Number of steps before evaluating PPO model (default: 1e5)' # PPO 模型评估的时间间隔（以时间步为单位）。默认值: 1×10^5 （10 万步）。
    )