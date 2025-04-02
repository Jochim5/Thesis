import numpy as np

'''
计算生成机器人中与模板体素类型一致的位置比例：
TypeSimilarity = Number of voxels matching type 3 in template / Total voxels
'''
#from evogym.examples.make_gifs import Robot


def calculate_type_similarity(robot, template, target_voxel=3):
    # 找到模板机器人中 target_voxel 的位置
    template_positions = (template == target_voxel)

    # 计算生成机器人在这些位置上是否也是 target_voxel
    matching_positions = (robot == target_voxel) & template_positions

    # 计算匹配比例
    total_target_voxels = template_positions.sum()  # 模板中 target_voxel 的总数
    if total_target_voxels == 0:
        return 0.0  # 如果模板中没有 target_voxel，相似度为 0
    similarity = matching_positions.sum() / total_target_voxels
    return similarity

'''
对比生成机器人与模板机器人在每个位置上是否完全一致：
Structure Similarity = Number of matching elements / Total elements
'''
def calculate_structure_similarity(robot, template):
    total_elements = robot.size
    matching_elements = (robot == template).sum()  # 完全匹配的体素数量
    similarity = matching_elements / total_elements
    return similarity

'''
将体素类型一致性和结构排列一致性结合，可以加权求和得到总相似度：
Total Similarity = 𝛼⋅Type Similarity + 𝛽⋅ Structure Similarity

其中：
α：控制体素类型的重要性；
β：控制结构匹配的重要性。
'''
def reward_function(test_robot, template_robot, target_voxel=3, alpha=0.5, beta=0.5):

    type_similarity = calculate_type_similarity(test_robot, template_robot, target_voxel)

    # 计算 Structure Similarity (结构相似度)
    structure_similarity = calculate_structure_similarity(test_robot, template_robot)

    # 计算 Total Similarity (总相似度)
    reward = alpha * type_similarity + beta * structure_similarity

    return reward


char_to_num = {
    'E': 0,
    'R': 1,
    'S': 2,
    'H': 3,
    'V': 4
}


# 转换 test_robot 字符串为二维数组
def convert_to_robot_format(test_robot, rows, cols):
    # 根据映射将字符转换为数字
    robot_array = np.array([char_to_num[char] for char in test_robot])

    # 重塑为二维数组，形状为 (rows, cols)
    return robot_array.reshape((rows, cols))

if __name__ == "__main__":
    test_robot = "HEHHHHHHRHHHEHHVHSESEREEH"
    #test_robot_array =np.array([
    #    [0., 3., 3., 0., 4.],
    #    [0., 3., 4., 3., 3.],
    #    [3., 3., 0., 3., 3.],
    #    [3., 4., 4., 3., 3.],
    #    [3., 3., 0., 1., 3.]
    #])

    template_robot = np.array([
        [3., 3., 3., 3., 3.],
        [3., 3., 3., 3., 3.],
        [3., 3., 0., 3., 3.],
        [3., 3., 0., 3., 3.],
        [3., 3., 0., 3., 3.]
    ])
    test_robot_array = convert_to_robot_format(test_robot, 5, 5)
    reward1 = reward_function(test_robot_array, template_robot, target_voxel=3, alpha=0.0, beta=1.0)
    reward2 = reward_function(test_robot_array, template_robot, target_voxel=3, alpha=0.5, beta=0.5)
    print(f"Reward: {reward1, reward2}")