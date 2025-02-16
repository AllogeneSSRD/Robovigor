import os

def check_unlabeled_images(directory):
    # 获取所有 JPG 和 PNG 图片
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    for image in image_files:
        txt = os.path.splitext(image)[0] + '.txt'
        if txt_files.count(txt) != 1:
            print(f'图片 {image} 没有对应的 TXT 文件或对应的 TXT 文件不唯一')


check_unlabeled_images("data\\characters")