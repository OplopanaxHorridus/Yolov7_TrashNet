import os
import random
import shutil

dataset_path = "C:/Users/dylan/OneDrive/Documents/WPI/Artificial_Intelligence/Final_Project_Yolo/yolov7-main/data/dataset-original"
train_path = "path_to_save_training_set"
test_path = "path_to_save_test_set"
train_ratio = 0.8  # Ratio of images to be included in the training set

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

#directories for training and test sets
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# go through each class folder
for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    images = os.listdir(class_folder)
    random.shuffle(images)

    # splitting htem training and test sets
    num_train = int(len(images) * train_ratio)
    train_images = images[:num_train]
    test_images = images[num_train:]

    #moving images to respective class folders in training and test sets
    for image in train_images:
        src_path = os.path.join(class_folder, image)
        dst_path = os.path.join(train_path, class_name, image)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

    for image in test_images:
        src_path = os.path.join(class_folder, image)
        dst_path = os.path.join(test_path, class_name, image)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
