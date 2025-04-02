import os

def merge_and_remove_duplicates(directory_path: str, output_file: str):
    # 存储所有机器人的结构
    robot_structures = set()

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith("generation_") and filename.endswith("_structures.txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                # 读取文件内容并去除重复行
                for line in file:
                    robot_structures.add(line.strip())  # strip() 去除行首尾空白字符

    # 将去重后的结构写入到custom_data.txt文件
    with open(output_file, "w") as outfile:
        for structure in robot_structures:
            outfile.write(structure + "\n")

        # 删除所有generation_i_structures.txt文件
        for filename in os.listdir(directory_path):
            if filename.startswith("generation_") and filename.endswith("_structures.txt"):
                os.remove(os.path.join(directory_path, filename))


if __name__ == "__main__":
    base_path = "C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\evogym\\examples\\saved_data\\hand_design_experiment"

    merge_and_remove_duplicates(base_path, os.path.join(base_path, "custom_data.txt"))