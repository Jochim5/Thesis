import os
import numpy as np
import shutil
import random
import math
import argparse
from typing import List

import evogym.envs
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, Structure
from evogym.world import WorldObject
from .compare_robot import *
from ga.robot_converter import load_robot_from_json



def run_compare_ga(
        args: argparse.Namespace,
):
    print()

    exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores = (
        args.exp_name,
        args.env_name,
        args.pop_size,
        args.structure_shape,
        args.max_evaluations,
        args.num_cores,
    )

    ### MANAGE DIRECTORIES ###
    home_path = os.path.join("saved_data", exp_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###

    is_continuing = False
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join("saved_data", exp_name, "metadata.txt")

        try:
            os.makedirs(os.path.join("saved_data", exp_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.close()

    else:
        temp_path = os.path.join("saved_data", exp_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' +
              f'max evals: {max_evaluations}.')

        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures: List[Structure] = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0

    # generate a population
    if not is_continuing:
        robot_path = "C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\evogym-design-tool-main\\evogym-design-tool-main\\src\\exported\\robot_Q.json"
        template_path = "C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\evogym-design-tool-main\\evogym-design-tool-main\\src\\exported\\best_walk_robot.json"
        temp_structure = load_robot_from_json(robot_path)
        template_robot, _ = load_robot_from_json(template_path)  # 只获取模板机器人的体素矩阵
        # 将第一个机器人添加到种群中
        structures.append(Structure(*temp_structure, 0))
        population_structure_hashes[hashable(temp_structure[0])] = True
        num_evaluations += 1

        for i in range(1, pop_size):
            mutated_structure = mutate(temp_structure[0].copy(), mutation_rate=0.5, num_attempts=150)
            while (hashable(mutated_structure[0]) in population_structure_hashes):
                mutated_structure = mutate(temp_structure[0].copy(), mutation_rate=0.5, num_attempts=150)
            if mutated_structure is not None:
                structures.append(Structure(*mutated_structure, i))
                population_structure_hashes[hashable(mutated_structure[0])] = True
                num_evaluations += 1

    # read status from file
    else:
        for g in range(start_gen + 1):
            for i in range(pop_size):
                save_path_structure = os.path.join("saved_data", exp_name, "generation_" + str(g), "structure",
                                                   str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(Structure(*structure_data, i))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen


    while True:

        ### UPDATE NUM SURVIORS ###
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))

        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join("saved_data", exp_name, "generation_" + str(generation), "structure")

        try:
            os.makedirs(save_path_structure)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### CALCULATE REWARD BASED ON SIMILARITY ###
        for structure in structures:
            robot_body = structure.body  # 获取当前机器人的体素矩阵
            reward = reward_function(robot_body, template_robot, target_voxel=3, alpha=0.5, beta=0.5)
            structure.reward = reward  # 将奖励值设置为机器人的适应度

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        # SAVE RANKING TO FILE
        temp_path = os.path.join("saved_data", exp_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

        ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival * 100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        # store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate=0.3, num_attempts=150)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children + num_survivors]

        generation += 1

        ### SAVE ROBOT STRUCTURE AS STRING ###
        # Write the robot structures to a text file
        structure_file_path = os.path.join("saved_data", exp_name, f"generation_{generation}_structures.txt")
        with open(structure_file_path, "a") as f:
            for structure in structures:
                robot_body = structure.body  # 获取机器人体素矩阵
                structure_string = ""

                # 遍历体素矩阵，将每个值转换为对应的字符
                for row in robot_body:
                    for value in row:
                        if value == 0:
                            structure_string += "E"
                        elif value == 1:
                            structure_string += "R"
                        elif value == 2:
                            structure_string += "S"
                        elif value == 3:
                            structure_string += "H"
                        elif value == 4:
                            structure_string += "V"

                # 写入转换后的结构字符串到文件
                f.write(structure_string + "\n")


