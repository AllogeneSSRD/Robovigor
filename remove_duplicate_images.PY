import os

test_images_folder = "G:\Code\YOLO\yolov5-7.0\data\images"
train_images_folder = "train"

"""jpg文件"""
test_images = [f for f in os.listdir(test_images_folder) if f.endswith('.jpg', 'png')]
train_images = [f for f in os.listdir(train_images_folder) if f.endswith('.jpg', 'png')]

"""删除训练集中与测试集重复的jpg文件"""
for train_image in train_images:
    if train_image in test_images:
        os.remove(os.path.join(train_images_folder, train_image))
        print(f"Deleted {train_image} from folder B.")