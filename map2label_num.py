import os
import re

character = {
    "nahida": ["0", "*"],
    # 其他字符映射...
}

directory = "data\\characters"

for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.txt'):
            if '_' in filename:
                name_part = filename.split('_')[0]
                if name_part in character:
                    number = character[name_part][0]
                    with open(os.path.join(root, filename), 'r') as file:
                        lines = file.readlines()

                    print(f"已修改文件: {filename} --- {lines[0].strip()} -> {number}")

                    # 修改第一行的第一个空格前的内容
                    lines[0] = re.sub(r'^\d+', number, lines[0])
                    with open(os.path.join(root, filename), 'w') as file:
                        file.writelines(lines)