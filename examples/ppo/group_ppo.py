import os
import numpy as np
import json
import shutil
import argparse

from ppo.run import run_ppo
from ppo.args import add_ppo_args
import utils.mp_group as mp
import evogym.envs
from evogym import WorldObject
#这段代码的主要作用是在多个环境中测试多个机器人，以获得它们的控制器（policy）以及对应的性能（奖励值）
class SimJob(): #simjob类 用于存储仿真任务的信息。


    def __init__(self, name, robots, envs):
        self.name = name # 仿真任务的名称。
        self.robots = robots # 一个包含机器人名称的列表。
        self.envs = envs # 一个包含环境名称的列表。

    def get_data(self,):
        return {'robots': self.robots, 'envs': self.envs} # 返回任务数据的字典格式。

class RunData(): # 用于存储每个机器人在特定环境下的运行结果。

    def __init__(self, robot, env, job_name):
        self.robot = robot
        self.env = env
        self.job_name = job_name
        self.reward = 0
    def set_reward(self, reward):
        print(f'setting reward for {self.robot} in {self.env}... {reward}') # 设置并打印奖励值。
        self.reward = reward

def read_robot_from_file(file_name): # 尝试从多个可能的路径读取机器人结构数据。参数 file_name: 机器人文件名。如果文件是 .json 格式，使用 WorldObject 提取结构和连接。如果文件是 .npz 格式，用 numpy.load 读取数据。返回值是一个元组 (structure, connections)。
    possible_paths = [
        os.path.join(file_name),
        os.path.join(f'{file_name}.npz'),
        os.path.join(f'{file_name}.json'),
        os.path.join('world_data', file_name),
        os.path.join('world_data', f'{file_name}.npz'),
        os.path.join('world_data', f'{file_name}.json'),
    ]

    best_path = None
    for path in possible_paths:
        if os.path.exists(path):
            best_path = path
            break

    if best_path.endswith('json'):
        robot_object = WorldObject.from_json(best_path)
        return (robot_object.get_structure(), robot_object.get_connections())
    if best_path.endswith('npz'):
        structure_data = np.load(best_path)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        return tuple(structure)
    return None

def clean_name(name): # 去掉文件名中的路径分隔符和扩展名，只保留文件的纯名称
    while name.find('/') != -1:
        name = name[name.find('/')+1:]
    while name.find('\\') != -1:
        name = name[name.find('\\')+1:]
    while name.find('.') != -1:
        name = name[:name.find('.')]
    return name

def run_group_ppo(experiment_name, sim_jobs): # experiment_name: 实验名称，用于文件组织。sim_jobs: 一个包含多个 SimJob 对象的列表，定义了所有要运行的任务。
    ### ARGS ### # 创建解析器并添加 PPO 的命令行参数。使用 args 存储所有解析结果。
    parser = argparse.ArgumentParser(description='Arguments for group PPO script')
    add_ppo_args(parser)
    args = parser.parse_args()

    ### MANAGE DIRECTORIES ### 创建实验目录
    exp_path = os.path.join("saved_data", experiment_name)
    try:
        os.makedirs(exp_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Delete and override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(exp_path)
            print()
        else:
            quit()

    ### RUN JOBS ###
    run_data = []
    group = mp.Group()
    out = {}

    for job in sim_jobs:

        out[job.name] = {}
        save_path_structure = os.path.join(exp_path, f'{job.name}', "structure")
        save_path_controller = os.path.join(exp_path, f'{job.name}', "controller") #为每个任务创建单独的目录：structure: 用于存储机器人结构。controller: 用于存储 PPO 模型。

        try:
            os.makedirs(save_path_structure) # 创建一个目录路径，存储机器人，如果路径的父目录不存在，也会被创建。
        except:
            pass

        try:
            os.makedirs(save_path_controller) # 创建一个目录路径，存储控制器，如果路径的父目录不存在，也会被创建。
        except:
            pass

        count = 0
        for env_name in job.envs: # 遍历当前任务的所有环境。
            out[job.name][env_name] = {} # 为每个环境初始化一个空字典，用于存储该环境中各机器人的结果。
            for robot_name in job.robots:  #遍历当前任务的所有机器人。
                out[job.name][env_name][robot_name] = 0 # 初始化机器人在当前环境中的奖励为 0。
                
                run_data.append(RunData(robot_name, env_name, job.name)) # 一个列表，用于存储所有运行任务的 RunData 对象，记录机器人、环境和任务名称。
                structure = read_robot_from_file(robot_name)
                
                temp_path = os.path.join(save_path_structure, f'{clean_name(robot_name)}_{env_name}.npz') # 生成机器人结构保存的文件路径，格式为 {cleaned_robot_name}_{env_name}.npz。
                np.savez(temp_path, structure[0], structure[1]) # 机器人结构数组和连接数组
                
                ppo_args = (args, structure[0], env_name, save_path_controller, f'{clean_name(robot_name)}_{env_name}', structure[1]) # ppo_args:args: PPO 的超参数。structure[0]: 机器人结构。env_name: 当前环境名称。save_path_controller: 控制器的保存路径。文件名: 使用清理后的机器人名称和环境名。structure[1]: 机器人连接关系。
                group.add_job(run_ppo, ppo_args, callback=run_data[-1].set_reward) # group.add_job: run_ppo: 要运行的函数。ppo_args: 函数所需的参数。callback=run_data[-1].set_reward: 在 run_ppo 完成后，调用 set_reward 方法更新奖励值。
                
    group.run_jobs(2) # 并行运行任务组，最多同时运行 2 个任务。

    ### SAVE RANKING TO FILE ##
    for data in run_data:
        out[data.job_name][data.env][data.robot] = data.reward

    out_file = os.path.join(exp_path, 'output.json')
    with open(out_file, 'w') as f:
        json.dump(out, f)