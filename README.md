Yolo_Code.ipynb is the main  code for the project. It includes image augmentation and data parsing. Follow the directions in between code snippets. It is also recommended to  run in google collab, and it is also recommended to upgrade to Google Pro if you are going to be doing a lot of optimization of the models. 
#Generates the YOLO dataset (since it needs labels). Make sure you code is in the same folder as Yolov7 main folder as well. The dataset used was TrashNet dataset, but other datasets could be used. Just make sure that it is setup the same as below with the dataset in the content folder.
yolo_dir = '/content/yolo_trash_net_data/'
label_dir = '/content/yolo_labels'
for folder in ['test', 'train', 'val']:

        os.rename(old_file, new_file)
