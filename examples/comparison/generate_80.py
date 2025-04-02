import random


def modify_and_convert(template, num_changes_range=(1, 5), filename="output.txt", num_samples=300):
    """
    生成 num_samples 个修改后的字符串，并转换为对应字母表示形式，写入文件。

    参数：
    - template (str)：模板字符串，例如 "3333333333330333303333033"
    - num_changes_range (tuple)：随机修改的字符数范围，默认为 (1, 5)
    - filename (str)：输出文件名，默认为 "output.txt"
    - num_samples (int)：生成的样本数量，默认为 300
    """

    mapping = {"0": "E", "1": "R", "2": "S", "3": "H", "4": "V"}  # 数字到字母的映射

    with open(filename, "w") as file:
        for i in range(num_samples):
            template_list = list(template)  # 转换为列表以便修改
            length = len(template_list)  # 获取字符串长度
            num_changes = random.randint(*num_changes_range)  # 选择修改的字符数

            change_indices = random.sample(range(length), num_changes)  # 随机选取修改位置

            for idx in change_indices:
                old_value = template_list[idx]
                new_value = str(random.choice([x for x in range(5) if str(x) != old_value]))  # 确保新值不同
                template_list[idx] = new_value  # 替换字符

            modified_string = "".join(template_list)  # 转换回字符串
            structure_string = "".join(mapping[c] for c in modified_string)  # 转换成字母字符串

            # 写入文件
            file.write(structure_string + "\n")


    print(f"{num_samples} 个修改后的字符串已写入 {filename}")


# 运行代码
template_str = "3333333333330333303333033"
modify_and_convert(template_str)