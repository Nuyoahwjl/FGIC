import os
import hashlib
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import imagehash
from PIL import Image

class DuplicateImageHandler:
    def __init__(self, dataset_path=None, csv_file=None, image_column='image', 
                 label_column=None, hash_size=8, threshold=5):
        """初始化重复图片处理器
        
        Args:
            dataset_path: 数据集根路径（方式1：处理文件夹下所有图片；方式2：与CSV中的相对路径拼接）
            csv_file: CSV文件路径（方式2：只处理CSV中指定的图片）
            image_column: CSV中图片路径列的名称，默认'image'
            label_column: CSV中标签列的名称，如果为None则从文件夹结构推断
            hash_size: 哈希大小，默认8（生成64位哈希）
            threshold: 汉明距离阈值，距离小于等于此值认为是重复图片
        """
        if dataset_path is None and csv_file is None:
            raise ValueError("必须提供 dataset_path 或 csv_file 中的一个")
        
        if csv_file is not None and dataset_path is None:
            raise ValueError("使用CSV模式时必须提供 dataset_path 用于构建完整的文件路径")
        
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.csv_file = csv_file
        self.image_column = image_column
        self.label_column = label_column
        self.hash_size = hash_size
        self.threshold = threshold
        
        self.image_hashes = []
        self.similar_groups = []
        self.images_to_keep = []
        self.images_to_remove = []
        self.input_mode = "folder" if csv_file is None else "csv"
        
        print(f"初始化完成，输入模式: {self.input_mode}")
        print(f"数据集路径: {self.dataset_path}")
        if self.input_mode == "csv":
            print(f"CSV文件: {self.csv_file}")
    
    def compute_image_hash(self, image_path, hash_type='phash'):
        """计算图像的感知哈希值"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if hash_type == 'phash':
                    return imagehash.phash(img, hash_size=self.hash_size)
                elif hash_type == 'ahash':
                    return imagehash.average_hash(img, hash_size=self.hash_size)
                elif hash_type == 'dhash':
                    return imagehash.dhash(img, hash_size=self.hash_size)
                elif hash_type == 'whash':
                    return imagehash.whash(img, hash_size=self.hash_size)
                else:
                    return imagehash.phash(img, hash_size=self.hash_size)
        except Exception as e:
            print(f"Error computing image hash for {image_path}: {e}")
            return None
    
    def get_image_files_from_folder(self):
        """从文件夹获取所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        image_files = []
        seen_files = set()  # 用于去重
        
        for file_path in self.dataset_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in image_extensions:
                    # 使用绝对路径作为唯一标识符避免重复
                    abs_path = file_path.resolve()
                    if abs_path not in seen_files:
                        seen_files.add(abs_path)
                        image_files.append(file_path)
        
        return image_files
        
    
    def get_image_files_from_csv(self):
        """从CSV文件获取图片文件信息"""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"CSV文件加载成功，共{len(df)}行数据")
            
            if self.image_column not in df.columns:
                raise ValueError(f"CSV文件中未找到图片列 '{self.image_column}'")
            
            # 检查标签列
            if self.label_column and self.label_column not in df.columns:
                raise ValueError(f"CSV文件中未找到标签列 '{self.label_column}'")
            
            # 检查是否提供了dataset_path
            if self.dataset_path is None:
                raise ValueError("处理CSV文件时必须提供 dataset_path 用于构建完整的文件路径")
            
            image_info = []
            missing_files = []
            
            for idx, row in df.iterrows():
                image_path_str = str(row[self.image_column]).strip()
                
                # 构建完整路径：dataset_path + CSV中的相对路径
                full_path = self.dataset_path / image_path_str
                
                # 检查文件是否存在
                if full_path.exists() and full_path.is_file():
                    # 获取标签
                    if self.label_column:
                        label = str(row[self.label_column])
                    else:
                        # 从CSV中的路径推断标签（第一级目录名）
                        # 例如: 000/0a046b7a1248450882dcd6eda90d7bdc.jpg -> 标签为 000
                        path_parts = Path(image_path_str).parts
                        if len(path_parts) > 1:
                            label = path_parts[0]  # 取第一级目录名
                        else:
                            # 如果没有目录结构，使用文件的父目录名
                            label = full_path.parent.name
                    
                    image_info.append({
                        'path': full_path,
                        'relative_path': image_path_str,
                        'label': label,
                        'csv_row': idx
                    })
                else:
                    missing_files.append(image_path_str)
            
            if missing_files:
                print(f"警告: 找不到以下 {len(missing_files)} 个文件:")
                for missing in missing_files[:10]:  # 只显示前10个
                    print(f"  - {missing}")
                if len(missing_files) > 10:
                    print(f"  ... 还有 {len(missing_files) - 10} 个文件")
            
            print(f"成功找到 {len(image_info)} 个图片文件")
            return image_info
            
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {e}")

    
    def get_image_label_from_folder(self, image_path):
        """从文件夹结构推断标签"""
        return image_path.parent.name
    
    def get_relative_path_from_folder(self, image_path):
        """获取相对于数据集根目录的路径"""
        try:
            return str(image_path.relative_to(self.dataset_path))
        except ValueError:
            return str(image_path)
    
    def scan_images(self, hash_type='phash'):
        """
        扫描所有图片并计算哈希值
        
        Args:
            hash_type: 哈希算法类型 ('phash', 'ahash', 'dhash', 'whash')
        """
        print(f"开始扫描图片并计算 {hash_type} 哈希值...")
        print(f"哈希大小: {self.hash_size}, 相似度阈值: {self.threshold}")
        
        if self.input_mode == "folder":
            # 方式1: 从文件夹扫描
            image_files = self.get_image_files_from_folder()
            print(f"在文件夹中找到 {len(image_files)} 个图片文件")
            
            for image_path in tqdm(image_files, desc="计算图片哈希值"):
                hash_value = self.compute_image_hash(image_path, hash_type)
                
                if hash_value is not None:
                    label = self.get_image_label_from_folder(image_path)
                    relative_path = self.get_relative_path_from_folder(image_path)
                    
                    self.image_hashes.append({
                        'path': image_path,
                        'relative_path': relative_path,
                        'label': label,
                        'hash': hash_value,
                        'hash_str': str(hash_value),
                        'processed': False,
                        'source': 'folder'
                    })
        
        else:
            # 方式2: 从CSV读取
            image_info_list = self.get_image_files_from_csv()
            
            for image_info in tqdm(image_info_list, desc="计算图片哈希值"):
                hash_value = self.compute_image_hash(image_info['path'], hash_type)
                
                if hash_value is not None:
                    self.image_hashes.append({
                        'path': image_info['path'],
                        'relative_path': image_info['relative_path'],
                        'label': image_info['label'],
                        'hash': hash_value,
                        'hash_str': str(hash_value),
                        'processed': False,
                        'source': 'csv',
                        'csv_row': image_info['csv_row']
                    })
        
        print(f"成功处理 {len(self.image_hashes)} 个图片文件")
    
    def find_similar_images(self):
        """使用汉明距离找到相似的图片"""
        print("使用汉明距离查找相似图片...")
        
        for i in range(len(self.image_hashes)):
            if self.image_hashes[i]['processed']:
                continue
                
            similar_group = [i]  # 存储索引
            
            # 与后续所有图片比较
            for j in range(i + 1, len(self.image_hashes)):
                if self.image_hashes[j]['processed']:
                    continue
                
                # 计算汉明距离
                hamming_distance = self.image_hashes[i]['hash'] - self.image_hashes[j]['hash']
                
                if hamming_distance <= self.threshold:
                    similar_group.append(j)
            
            # 标记为已处理
            for idx in similar_group:
                self.image_hashes[idx]['processed'] = True
            
            if len(similar_group) > 1:
                # 找到相似图片组
                group_images = [self.image_hashes[idx] for idx in similar_group]
                self.similar_groups.append(group_images)
                
                # 计算组内最大距离
                max_distance = 0
                base_hash = group_images[0]['hash']
                for img in group_images[1:]:
                    distance = base_hash - img['hash']
                    max_distance = max(max_distance, distance)
                
                labels = set(img['label'] for img in group_images)
                print(f"找到相似图片组: {len(similar_group)} 张图片, 标签: {labels}, 最大距离: {max_distance}")
        
        return self.similar_groups
    
    def process_duplicates(self):
        """处理重复/相似图片，确定哪些要保留，哪些要删除"""
        self.find_similar_images()
        
        print(f"共找到 {len(self.similar_groups)} 组相似图片")
        
        # 创建一个集合来追踪已处理的图片路径
        processed_paths = set()
        
        # 处理相似图片组
        for group_id, group in enumerate(self.similar_groups, 1):
            # 收集所有唯一标签
            unique_labels = set(img['label'] for img in group)
            
            # 计算组内的最大汉明距离
            max_distance = 0
            if len(group) > 1:
                base_hash = group[0]['hash']
                for img in group[1:]:
                    distance = base_hash - img['hash']
                    max_distance = max(max_distance, distance)
            
            if len(unique_labels) == 1:
                # 策略1: 相同标签，只保留一张（选择文件名最短的或第一张）
                kept_image = min(group, key=lambda x: len(x['relative_path']))
                
                keep_record = {
                    'relative_path': kept_image['relative_path'],
                    'label': kept_image['label'],
                    'reason': 'kept_from_same_label_duplicates',
                    'duplicate_group': f'group_{group_id}',
                    'hash': kept_image['hash_str'],
                    'max_group_distance': max_distance
                }
                
                # 如果是CSV模式，添加行号
                if self.input_mode == "csv":
                    keep_record['csv_row'] = kept_image['csv_row']
                
                self.images_to_keep.append(keep_record)
                processed_paths.add(str(kept_image['path']))
                
                # 其他图片加入删除列表
                for img in group:
                    if str(img['path']) != str(kept_image['path']):
                        remove_record = {
                            'relative_path': img['relative_path'],
                            'label': img['label'],
                            'reason': 'duplicate_same_label',
                            'duplicate_group': f'group_{group_id}',
                            'hash': img['hash_str'],
                            'max_group_distance': max_distance,
                            'distance_to_kept': kept_image['hash'] - img['hash']
                        }
                        
                        if self.input_mode == "csv":
                            remove_record['csv_row'] = img['csv_row']
                        
                        self.images_to_remove.append(remove_record)
                        processed_paths.add(str(img['path']))
                
                print(f"组 {group_id}: 保留1张，删除{len(group)-1}张 (标签: '{list(unique_labels)[0]}', 最大距离: {max_distance})")
            
            else:
                # 策略2: 不同标签，删除所有相似图片
                for img in group:
                    remove_record = {
                        'relative_path': img['relative_path'],
                        'label': img['label'],
                        'reason': 'duplicate_different_labels',
                        'duplicate_group': f'group_{group_id}',
                        'hash': img['hash_str'],
                        'max_group_distance': max_distance
                    }
                    
                    if self.input_mode == "csv":
                        remove_record['csv_row'] = img['csv_row']
                    
                    self.images_to_remove.append(remove_record)
                    processed_paths.add(str(img['path']))
                
                print(f"组 {group_id}: 删除所有{len(group)}张 (冲突标签: {unique_labels}, 最大距离: {max_distance})")
        
        # 添加所有未被处理的图片到保留列表（唯一图片）
        for img in self.image_hashes:
            if str(img['path']) not in processed_paths:
                keep_record = {
                    'relative_path': img['relative_path'],
                    'label': img['label'],
                    'reason': 'unique',
                    'duplicate_group': None,
                    'hash': img['hash_str']
                }
                
                if self.input_mode == "csv":
                    keep_record['csv_row'] = img['csv_row']
                
                self.images_to_keep.append(keep_record)
    
    def export_results(self, keep_csv="images_to_keep.csv", remove_csv="images_to_remove.csv", 
                      summary_csv="duplicate_summary.csv"):
        """导出结果到CSV文件"""
        # 导出要保留的图片列表
        if self.images_to_keep:
            keep_df = pd.DataFrame(self.images_to_keep)
            keep_df.to_csv(keep_csv, index=False)
            print(f"导出 {len(self.images_to_keep)} 张要保留的图片到 '{keep_csv}'")
        
        # 导出要删除的图片列表
        if self.images_to_remove:
            remove_df = pd.DataFrame(self.images_to_remove)
            remove_df.to_csv(remove_csv, index=False)
            print(f"导出 {len(self.images_to_remove)} 张要删除的图片到 '{remove_csv}'")
        
        # 创建摘要信息
        summary_data = []
        
        # 统计信息
        total_images = len(self.image_hashes)
        unique_images = len([img for img in self.images_to_keep if img['reason'] == 'unique'])
        same_label_groups = 0
        diff_label_groups = 0
        
        for group in self.similar_groups:
            unique_labels = set(img['label'] for img in group)
            if len(unique_labels) == 1:
                same_label_groups += 1
            else:
                diff_label_groups += 1
        
        summary_data.extend([
            {
                'metric': 'input_mode',
                'value': self.input_mode,
                'description': 'Input method: folder or csv'
            },
            {
                'metric': 'total_images_scanned',
                'value': total_images,
                'description': 'Total number of images processed'
            },
            {
                'metric': 'hash_algorithm',
                'value': 'imagehash',
                'description': 'Hash algorithm used'
            },
            {
                'metric': 'hash_size',
                'value': self.hash_size,
                'description': 'Hash size parameter'
            },
            {
                'metric': 'similarity_threshold',
                'value': self.threshold,
                'description': 'Hamming distance threshold for similarity'
            },
            {
                'metric': 'unique_images',
                'value': unique_images,
                'description': 'Images with no similar matches'
            },
            {
                'metric': 'images_to_keep',
                'value': len(self.images_to_keep),
                'description': 'Total images that will be kept'
            },
            {
                'metric': 'images_to_remove',
                'value': len(self.images_to_remove),
                'description': 'Total images that will be removed'
            },
            {
                'metric': 'similar_groups_total',
                'value': len(self.similar_groups),
                'description': 'Total groups of similar images'
            },
            {
                'metric': 'same_label_similar_groups',
                'value': same_label_groups,
                'description': 'Similar groups with same labels'
            },
            {
                'metric': 'different_label_similar_groups',
                'value': diff_label_groups,
                'description': 'Similar groups with different labels'
            }
        ])
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv, index=False)
        print(f"导出摘要信息到 '{summary_csv}'")
    
    def validate_results(self):
        """验证结果的正确性"""
        print("验证结果...")
        
        keep_paths = set(img['relative_path'] for img in self.images_to_keep)
        remove_paths = set(img['relative_path'] for img in self.images_to_remove)
        all_paths = set(img['relative_path'] for img in self.image_hashes)
        
        # 检查重复
        overlapping = keep_paths.intersection(remove_paths)
        if overlapping:
            print(f"错误: 发现 {len(overlapping)} 张图片同时在保留和删除列表中:")
            for path in list(overlapping)[:5]:
                print(f"  - {path}")
            return False
        
        # 检查完整性
        processed_paths = keep_paths.union(remove_paths)
        missing = all_paths - processed_paths
        if missing:
            print(f"错误: 发现 {len(missing)} 张图片未被处理:")
            for path in list(missing)[:5]:
                print(f"  - {path}")
            return False
        
        extra = processed_paths - all_paths
        if extra:
            print(f"错误: 发现 {len(extra)} 张多余的图片:")
            for path in list(extra)[:5]:
                print(f"  - {path}")
            return False
        
        print("✓ 验证通过: 保留和删除列表无重复")
        print(f"✓ 总图片数: {len(all_paths)}")
        print(f"✓ 保留图片数: {len(keep_paths)}")
        print(f"✓ 删除图片数: {len(remove_paths)}")
        print(f"✓ 总和: {len(keep_paths) + len(remove_paths)} (与总数匹配)")
        
        return True
    
    def generate_report(self, output_file=None):
        """生成处理报告"""
        report = f"""
重复图片处理报告 (使用 ImageHash)
=================================

配置信息:
- 输入模式: {self.input_mode}
- 哈希算法: 感知哈希 (phash)
- 哈希大小: {self.hash_size}
- 相似度阈值 (汉明距离): {self.threshold}

处理结果:
- 扫描图片总数: {len(self.image_hashes)}
- 保留图片数: {len(self.images_to_keep)}
- 删除图片数: {len(self.images_to_remove)}
- 相似图片组数: {len(self.similar_groups)}

详细统计:
- 唯一图片: {len([img for img in self.images_to_keep if img['reason'] == 'unique'])}
- 相同标签组保留的图片: {len([img for img in self.images_to_keep if img['reason'] == 'kept_from_same_label_duplicates'])}
- 相同标签组删除的图片: {len([img for img in self.images_to_remove if img['reason'] == 'duplicate_same_label'])}
- 不同标签组删除的图片: {len([img for img in self.images_to_remove if img['reason'] == 'duplicate_different_labels'])}

导出文件:
- images_to_keep.csv: 要保留的图片列表 (含哈希值)
- images_to_remove.csv: 要删除的图片列表 (含相似度距离)
- duplicate_summary.csv: 摘要统计信息

注意: 每张图片只出现在一个列表中 (保留或删除)。
汉明距离表示图片相似程度 (0 = 完全相同, 数值越大越不相似)
"""
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        print(report)

# 使用示例
if __name__ == "__main__":
    
    # 示例: CSV文件包含图片路径和标签
    handler = DuplicateImageHandler(
        dataset_path="./data",  # 数据集根路径
        #csv_file="images.csv",            # CSV文件路径
        #image_column="image",                 # 图片路径列名
        hash_size=8,
        threshold=5
    )
    
    # 执行处理
    try:
        # 扫描图片并计算哈希
        handler.scan_images(hash_type='phash')
        
        # 处理重复图片
        handler.process_duplicates()
        
        # 验证结果
        if handler.validate_results():
            # 导出结果
            handler.export_results(
                keep_csv="images_to_keep.csv",
                remove_csv="images_to_remove.csv", 
                summary_csv="duplicate_summary.csv"
            )
            
            # 生成报告
            handler.generate_report("duplicate_processing_report.txt")
            
            print("\n" + "="*50)
            print("处理完成!")
            print("已通过验证，每张图片只出现在一个列表中。")
        else:
            print("结果验证失败，请检查代码逻辑。")
    
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
