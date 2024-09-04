import os
import yaml
import cv2
import base64

def gen_only_yolo_folder(total_folder_path,detection_yolo):
    
    # # 建立大資料夾包含各個格式的資料夾跟zip檔
    # os.makedirs(total_folder_path)
    # print(f'\nCreating main folder：{total_folder_path}\n')

    # # 建立Yolo資料集
    # os.makedirs(detection_yolo)
    # print(f'Creating Yolo folder：{detection_yolo}')
    # yolo_train = detection_yolo+'/datasets'
    # os.makedirs(yolo_train)
    # # print(f'已創建YOLO格式所需的資料夾train：{yolo_train}')
    # yolo_train_images = yolo_train+'/images'
    # os.makedirs(yolo_train_images)
    # # print(f'已創建train所需的資料夾images：{yolo_train_images}')
    # yolo_train_labels = yolo_train+'/labels'
    # os.makedirs(yolo_train_labels)
    # # print(f'已創建train所需的資料夾labels：{yolo_train_labels}\n')

    # Create main folder containing subfolders and zip files
    os.makedirs(total_folder_path)
    print(f'\nCreating main folder: {total_folder_path}\n')

    # Create YOLO dataset folder
    os.makedirs(detection_yolo)
    print(f'Creating YOLO folder: {detection_yolo}')
    yolo_train = detection_yolo + '/datasets'
    os.makedirs(yolo_train)
    print(f'Created YOLO format required folder train: {yolo_train}')
    yolo_train_images = yolo_train + '/images'
    os.makedirs(yolo_train_images)
    print(f'Created train required folder images: {yolo_train_images}')
    yolo_train_labels = yolo_train + '/labels'
    os.makedirs(yolo_train_labels)
    print(f'Created train required folder labels: {yolo_train_labels}\n')


def genAllDatayaml(output_file, names):
    # caculating nc (classes count)
    nc = len(names)
    
    # define yaml format
    data = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': nc,
        'names': names
    }

    # define output path
    output_file = output_file + '/data.yaml'

    # Use YAML SafeDumper to ensure Order
    class MyDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow, False)

    # Construct YAML string with proper indentation
    yaml_content = (
        f"train: {data['train']}\n"
        f"val: {data['val']}\n"
        f"test: {data['test']}\n\n"
        f"nc: {data['nc']}\n"
        f"names: {data['names']}\n"
    )

    # wirte YAML file
    with open(output_file, 'w') as f:
        f.write(yaml_content)

    print(f"Create YAML File Required for Roboflow YOLO '{output_file}'\n")

def resize_frame(frame, height):
    """Adjust the size of the image to the specified height, keeping the aspect ratio."""
    aspect_ratio = frame.shape[1] / frame.shape[0]
    width = int(height * aspect_ratio)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized_frame

def encode_image_to_base64(image):
    """Encode an image to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

import os
import shutil
import random

def split_dataset(source_folder, train_ratio, valid_ratio):
    train_ratio = float(train_ratio)
    valid_ratio = float(valid_ratio)

    # Check if the folders exist
    images_folder = os.path.join(source_folder, 'datasets/images')
    labels_folder = os.path.join(source_folder, 'datasets/labels')

    if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
        raise FileNotFoundError("Images or labels folder does not exist in the source directory.")

    
    # Create output folders
    os.makedirs(os.path.join(source_folder, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(source_folder, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(source_folder, 'valid/images'), exist_ok=True)
    os.makedirs(os.path.join(source_folder, 'valid/labels'), exist_ok=True)
    os.makedirs(os.path.join(source_folder, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(source_folder, 'test/labels'), exist_ok=True)

    # Get all image file names
    images = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
    # Shuffle the file names list
    random.shuffle(images) 

    # Calculate the number of files for each dataset
    total_count = len(images)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)

    # Distribute files to datasets
    train_images = images[:train_count]
    valid_images = images[train_count:train_count + valid_count]
    test_images = images[train_count + valid_count:]

    def move_files(image_list, source_images_folder, source_labels_folder, dest_folder):
        for image_file in image_list:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            # moving picture
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(dest_folder, 'images', image_file))
            # moving label
            if os.path.exists(os.path.join(source_labels_folder, label_file)):
                shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(dest_folder, 'labels', label_file))

    move_files(train_images, images_folder, labels_folder, os.path.join(source_folder, 'train'))
    move_files(valid_images, images_folder, labels_folder, os.path.join(source_folder, 'valid'))
    move_files(test_images, images_folder, labels_folder, os.path.join(source_folder, 'test'))
    # delete origin folder
    datasets_folder = os.path.join(source_folder, 'datasets')
    if os.path.exists(datasets_folder):
        shutil.rmtree(datasets_folder)
    print("Data splitting complete!")

