import os
from PIL import Image
from torch.utils.data import Dataset
from .augmentation import HighNoiseFineGrainedAugmentation

def safe_convert_to_rgb(image_path):
    """安全地将图像转换为RGB模式，处理所有可能的模式"""
    try:
        image = Image.open(image_path)
        
        # 处理所有可能的图像模式
        if image.mode in ('P', 'PA', 'RGBA', 'LA', 'RGBa'):
            # 对于调色板和带透明度的图像，转换为RGBA然后合并到白色背景
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # 创建白色背景并合并
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            # 其他模式直接转换为RGB
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # 返回一个默认的RGB图像作为后备
        return Image.new('RGB', (299, 299), (255, 255, 255))

class MyDataset(Dataset):
    def __init__(self, root_dir, num_class, mode='train', resize_scale=(0.4, 1.0)):
        self.root_dir = root_dir
        self.mode = mode
        self.transform_generator = HighNoiseFineGrainedAugmentation(use_strong_aug=True)
        self.train_transform = self.transform_generator.get_train_transforms(resize_scale=resize_scale)
        self.valid_transform = self.transform_generator.get_val_transforms()
        self.dataset_imgs = []
        self.dataset_labels = {}

        for c in range(num_class):
        # 使用格式化字符串添加前导零，确保文件夹名称为000, 001, ...
            if num_class >= 1000:
                class_folder = f"{c:04d}"
            else:
                class_folder = f"{c:03d}"
            class_path = os.path.join(self.root_dir , class_folder)
            
            imgs = os.listdir(class_path)
            for img in imgs:
                img_path=os.path.join(class_folder,img)
                full_img_path = os.path.join(self.root_dir, img_path)
                # 验证图像是否可以正常打开
                try:
                    # with Image.open(full_img_path) as img_file:
                    #     img_file.verify()  # 验证图像完整性
                    self.dataset_imgs.append(img_path)
                    self.dataset_labels[img_path]=c
                except Exception as e:
                    print(f"Invalid image skipped: {full_img_path}, Error: {e}")
        
        print(f"Dataset initialized with {len(self.dataset_imgs)} images")
                
    def __getitem__(self, index):
        img_path = self.dataset_imgs[index]
        target = self.dataset_labels[img_path]
        image = safe_convert_to_rgb(os.path.join(self.root_dir, img_path))   
        if self.mode == 'train':
            img = self.train_transform(image)
        elif self.mode == 'valid':
            img = self.valid_transform(image)
        return img, target
    
    def __len__(self):
        return len(self.dataset_imgs)