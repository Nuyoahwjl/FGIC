#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能重复图片检测和清理工具
支持多种哈希算法、多种选择策略、多进程处理
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("请安装必要的依赖包:")
    print("pip install Pillow imagehash pandas tqdm")
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("建议安装numpy以获得更好的性能: pip install numpy")


def compute_single_hash(args):
    """计算单个图片的哈希值（用于多进程）"""
    image_path, hash_type, hash_size = args
    try:
        with Image.open(image_path) as img:
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            if hash_type == 'phash':
                return imagehash.phash(img, hash_size)
            elif hash_type == 'ahash':
                return imagehash.average_hash(img, hash_size)
            elif hash_type == 'dhash':
                return imagehash.dhash(img, hash_size)
            elif hash_type == 'whash':
                return imagehash.whash(img, hash_size)
            else:
                return imagehash.phash(img, hash_size)
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None


class DuplicateImageCleaner:
    def __init__(self, dataset_path=None, csv_file=None, image_column='image', 
                 label_column=None, hash_size=8, threshold=5):
        """
        初始化重复图片清理器
        
        Args:
            dataset_path: 数据集根目录路径（字符串或Path对象）
            csv_file: CSV文件路径（可选，用于指定图片列表）
            image_column: CSV中图片路径列的名称
            label_column: CSV中标签列的名称（可选）
            hash_size: 哈希大小（默认8，生成64位哈希）
            threshold: 汉明距离阈值（小于等于此值认为是相似图片）
        """
        # 路径处理
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.csv_file = csv_file
        self.image_column = image_column
        self.label_column = label_column
        
        # 参数设置
        self.hash_size = hash_size
        self.threshold = threshold
        
        # 确定输入模式
        if csv_file:
            self.input_mode = "csv"
            if not Path(csv_file).exists():
                raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
        elif dataset_path:
            self.input_mode = "folder"
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
        else:
            raise ValueError("必须提供 dataset_path 或 csv_file 参数")
        
        # 数据存储
        self.image_hashes = []
        self.similar_groups = []
        self.images_to_keep = []
        self.images_to_remove = []
        
        print(f"初始化完成:")
        print(f"  输入模式: {self.input_mode}")
        print(f"  数据集路径: {self.dataset_path}")
        print(f"  CSV文件: {self.csv_file}")
        print(f"  哈希大小: {self.hash_size}x{self.hash_size}")
        print(f"  相似度阈值: {self.threshold}")

    def get_image_files_from_folder(self):
        """从文件夹获取所有图片文件（优化版）"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # 使用生成器表达式和集合操作优化
        image_files = [
            file_path for file_path in self.dataset_path.rglob('*')
            if file_path.is_file() and file_path.suffix.lower() in image_extensions
        ]
        
        # 去重（使用resolve()获得绝对路径）
        seen = set()
        unique_files = []
        for file_path in image_files:
            abs_path = file_path.resolve()
            if abs_path not in seen:
                seen.add(abs_path)
                unique_files.append(file_path)
        
        return unique_files

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
        """从文件夹路径推断图片标签"""
        try:
            relative_path = image_path.relative_to(self.dataset_path)
            return relative_path.parent.name
        except:
            return "unknown"

    def get_relative_path_from_folder(self, image_path):
        """获取相对于数据集根目录的相对路径"""
        try:
            return str(image_path.relative_to(self.dataset_path))
        except:
            return str(image_path)

    def scan_images(self, hash_type='phash', use_multiprocessing=True, num_workers=None):
        """
        扫描所有图片并计算哈希值（优化版）
        
        Args:
            hash_type: 哈希算法类型
            use_multiprocessing: 是否使用多进程
            num_workers: 工作进程数，None表示自动选择
        """
        print(f"开始扫描图片并计算 {hash_type} 哈希值...")
        
        if self.input_mode == "folder":
            image_files = self.get_image_files_from_folder()
            image_info_list = [{'path': path, 'relative_path': self.get_relative_path_from_folder(path), 
                               'label': self.get_image_label_from_folder(path)} for path in image_files]
        else:
            image_info_list = self.get_image_files_from_csv()
        
        print(f"找到 {len(image_info_list)} 个图片文件")
        
        if not image_info_list:
            return
        
        # 准备参数
        hash_args = [(info['path'], hash_type, self.hash_size) for info in image_info_list]
        
        if use_multiprocessing and len(image_info_list) > 50:  # 图片较多时才使用多进程
            # 多进程处理
            if num_workers is None:
                num_workers = min(mp.cpu_count(), len(image_info_list))
            
            print(f"使用 {num_workers} 个进程并行计算哈希值...")
            
            with mp.Pool(num_workers) as pool:
                hash_results = list(tqdm(
                    pool.imap(compute_single_hash, hash_args),
                    total=len(hash_args),
                    desc="计算图片哈希值"
                ))
        else:
            # 单进程处理
            print("使用单进程计算哈希值...")
            hash_results = []
            for args in tqdm(hash_args, desc="计算图片哈希值"):
                hash_results.append(compute_single_hash(args))
        
        # 整理结果
        for i, (info, hash_value) in enumerate(zip(image_info_list, hash_results)):
            if hash_value is not None:
                self.image_hashes.append({
                    'path': info['path'],
                    'relative_path': info['relative_path'],
                    'label': info['label'],
                    'hash': hash_value,
                    'hash_str': str(hash_value),
                    'processed': False,
                    'source': self.input_mode,
                    'csv_row': info.get('csv_row')
                })
        
        print(f"成功处理 {len(self.image_hashes)} 个图片文件")
    def _build_similarity_groups_from_pairs(self, similar_pairs):
        """高性能版本 - 适合大规模数据"""
        if not similar_pairs:
            self.similar_groups = []
            return
        
        print(f"高性能构建相似组: {len(similar_pairs)} 个相似对...")
        
        # 使用numpy优化的Union-Find
        import numpy as np
        
        # 获取所有涉及的节点
        all_nodes = set()
        for i, j in similar_pairs:
            all_nodes.update([i, j])
        
        # 创建节点到连续索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        n_nodes = len(all_nodes)
        
        # 使用numpy数组的Union-Find
        parent = np.arange(n_nodes, dtype=np.int32)
        rank = np.zeros(n_nodes, dtype=np.int32)
        
        def find_vectorized(x):
            """向量化的find操作"""
            root = x
            while parent[root] != root:
                root = parent[root]
            
            # 路径压缩
            while parent[x] != root:
                next_x = parent[x]
                parent[x] = root
                x = next_x
            
            return root
        
        # 批量执行union操作
        for i, j in tqdm(similar_pairs, desc="执行Union操作"):
            idx_i = node_to_idx[i]
            idx_j = node_to_idx[j]
            
            root_i = find_vectorized(idx_i)
            root_j = find_vectorized(idx_j)
            
            if root_i != root_j:
                if rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                elif rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1
        
        # 收集连通分量
        from collections import defaultdict
        groups = defaultdict(list)
        
        for idx in range(n_nodes):
            root = find_vectorized(idx)
            original_node = idx_to_node[idx]
            groups[root].append(original_node)
        
        # 构建最终的相似图片组
        self.similar_groups = []
        total_images_in_groups = 0
        
        for group_nodes in groups.values():
            if len(group_nodes) >= 2:
                group = [self.image_hashes[i] for i in group_nodes]
                # 按文件名排序保证一致性
                group.sort(key=lambda x: x['relative_path'])
                
                # 重置processed标志
                for img in group:
                    img['processed'] = False
                
                self.similar_groups.append(group)
                total_images_in_groups += len(group)
        
        # 按组大小排序
        self.similar_groups.sort(key=len, reverse=True)
        
        print(f"构建完成:")
        print(f"  相似组数量: {len(self.similar_groups)}")
        print(f"  涉及图片: {total_images_in_groups}/{len(self.image_hashes)}")
        print(f"  重复率: {total_images_in_groups/len(self.image_hashes)*100:.1f}%")

    def find_similar_images_sorted(self):
        """使用排序优化查找相似图片（时间复杂度O(n log n)）"""
        print("使用排序优化查找相似图片...")
        
        # 将哈希值转换为可比较的整数形式
        sorted_hashes = sorted(
            enumerate(self.image_hashes),
            key=lambda x: int(str(x[1]['hash']), 16) if hasattr(x[1]['hash'], '__str__') else x[1]['hash']
        )
        
        similar_pairs = []
        threshold = self.threshold
        
        # 遍历排序后的哈希值，只比较附近的哈希值
        for i in range(len(sorted_hashes) - 1):
            idx_i, img_info_i = sorted_hashes[i]
            current_hash = img_info_i['hash']
            
            # 确保current_hash是字符串格式
            if hasattr(current_hash, '__str__'):
                current_hash = str(current_hash)
            
            # 只需要比较后面几个元素，设置一个合理的窗口大小
            for j in range(i + 1, min(i + 100, len(sorted_hashes))):
                idx_j, img_info_j = sorted_hashes[j]
                next_hash = img_info_j['hash']
                
                # 确保next_hash是字符串格式
                if hasattr(next_hash, '__str__'):
                    next_hash = str(next_hash)
                
                # 如果哈希值差异太大，后面的都不可能匹配，可以提前退出
                hash_i_int = int(current_hash, 16) if isinstance(current_hash, str) else current_hash
                hash_j_int = int(next_hash, 16) if isinstance(next_hash, str) else next_hash
                
                # 计算海明距离
                hamming_dist = bin(hash_i_int ^ hash_j_int).count('1')
                
                if hamming_dist <= threshold:
                    similar_pairs.append((idx_i, idx_j))
        
        # 构建相似组
        self._build_similarity_groups_from_pairs(similar_pairs)
        return similar_pairs

    def find_similar_images(self):
        """查找相似图片组（优化版）"""
        print(f"开始查找相似图片，阈值: {self.threshold}")
        
        if not self.image_hashes:
            print("没有图片需要处理")
            return
        
        n = len(self.image_hashes)
        print(f"需要处理 {n} 张图片")
        
        # 使用并查集来高效管理相似图片组
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # 路径压缩
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 使用numpy加速汉明距离计算
        global NUMPY_AVAILABLE
        if NUMPY_AVAILABLE and n > 100:
            try:
                # 将所有哈希值转换为numpy数组
                hash_size_bits = self.hash_size * self.hash_size
                hash_matrix = np.zeros((n, hash_size_bits), dtype=np.uint8)
                
                for i, img in enumerate(self.image_hashes):
                    hash_int = int(str(img['hash']), 16) if isinstance(img['hash'], str) else int(img['hash'])
                    # 转换为二进制位数组
                    binary_str = format(hash_int, f'0{hash_size_bits}b')
                    hash_matrix[i] = [int(bit) for bit in binary_str]
                
                # 批量计算汉明距离
                print("使用numpy批量计算汉明距离...")
                chunk_size = min(1000, n)  # 分块处理避免内存溢出
                
                for i in tqdm(range(0, n, chunk_size), desc="查找相似图片"):
                    end_i = min(i + chunk_size, n)
                    for j in range(i, n):
                        if j >= end_i:
                            # 计算chunk与单个元素的距离
                            distances = np.sum(hash_matrix[i:end_i] ^ hash_matrix[j], axis=1)
                            for idx, distance in enumerate(distances):
                                if distance <= self.threshold:
                                    union(i + idx, j)
                        elif i != j and j < end_i:
                            # chunk内部的距离计算
                            distance = np.sum(hash_matrix[i] ^ hash_matrix[j])
                            if distance <= self.threshold:
                                union(i, j)
                
            except Exception as e:
                print(f"numpy计算失败，回退到标准方法: {e}")
                NUMPY_AVAILABLE = False
        
        if not NUMPY_AVAILABLE or n <= 100:
            # 回退到原始方法但优化循环
            print("使用标准方法计算汉明距离...")
            
            # 预计算所有哈希值
            hash_values = []
            for img in self.image_hashes:
                if isinstance(img['hash'], str):
                    hash_values.append(int(img['hash'], 16))
                else:
                    # 对于ImageHash对象，先转换为字符串再转换为整数
                    hash_values.append(int(str(img['hash']), 16))
            
            # 优化的汉明距离计算
            def fast_hamming_distance(hash1, hash2):
                return bin(hash1 ^ hash2).count('1')
            
            # 使用三角矩阵避免重复计算
            for i in tqdm(range(n), desc="查找相似图片"):
                for j in range(i + 1, n):
                    distance = fast_hamming_distance(hash_values[i], hash_values[j])
                    if distance <= self.threshold:
                        union(i, j)
        
        # 收集相似图片组
        groups = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups[root].append(i)
        
        # 过滤掉只有一张图片的组
        self.similar_groups = []
        for group_indices in groups.values():
            if len(group_indices) > 1:
                group = [self.image_hashes[i] for i in group_indices]
                self.similar_groups.append(group)
        
        print(f"找到 {len(self.similar_groups)} 个相似图片组")

    def select_images_to_keep(self, strategy='largest_file', multi_label_strategy='remove_all'):
        """
        选择要保留的图片（支持多标签处理策略）
        
        Args:
            strategy: 同标签组内的选择策略
            multi_label_strategy: 多标签处理策略
                - 'remove_all': 若重复图片有多个不同标签，则全部删除
                - 'keep_one': 若重复图片有多个不同标签，仍保留一张（原有行为）
        """
        print(f"使用策略 '{strategy}' 选择要保留的图片...")
        print(f"多标签处理策略: '{multi_label_strategy}'")
        
        self.images_to_keep = []
        self.images_to_remove = []
        
        # 预计算所有文件信息（批量操作）
        if strategy in ['largest_file', 'smallest_file']:
            print("预计算文件大小...")
            file_sizes = {}
            for img in tqdm(self.image_hashes, desc="获取文件大小"):
                try:
                    file_sizes[img['relative_path']] = img['path'].stat().st_size
                except:
                    file_sizes[img['relative_path']] = 0
        elif strategy in ['highest_resolution', 'lowest_resolution']:
            print("预计算图片分辨率...")
            resolutions = {}
            for img in tqdm(self.image_hashes, desc="获取图片分辨率"):
                try:
                    with Image.open(img['path']) as pil_img:
                        resolutions[img['relative_path']] = pil_img.size[0] * pil_img.size[1]
                except:
                    resolutions[img['relative_path']] = 0
        
        # 统计多标签组信息
        multi_label_groups = 0
        same_label_groups = 0
        all_removed_groups = 0
        
        # 处理相似图片组
        for group in tqdm(self.similar_groups, desc="处理相似图片组"):
            # 分析组内标签分布
            labels_in_group = set(img['label'] for img in group)
            
            if len(labels_in_group) == 1:
                # 情况1: 重复图片标签相同，只留一张
                same_label_groups += 1
                
                if strategy == 'first':
                    keep_img = group[0]
                elif strategy == 'largest_file':
                    keep_img = max(group, key=lambda x: file_sizes.get(x['relative_path'], 0))
                elif strategy == 'smallest_file':
                    keep_img = min(group, key=lambda x: file_sizes.get(x['relative_path'], float('inf')))
                elif strategy == 'highest_resolution':
                    keep_img = max(group, key=lambda x: resolutions.get(x['relative_path'], 0))
                elif strategy == 'lowest_resolution':
                    keep_img = min(group, key=lambda x: resolutions.get(x['relative_path'], float('inf')))
                else:
                    keep_img = group[0]
                
                # 标记处理状态
                keep_img['processed'] = True
                keep_img['reason'] = f'kept_from_same_label_group_{len(group)}_images'
                self.images_to_keep.append(keep_img)
                
                # 其余图片标记为删除
                for img in group:
                    if img != keep_img:
                        img['processed'] = True
                        img['reason'] = f'removed_duplicate_same_label'
                        self.images_to_remove.append(img)
            
            else:
                # 情况2: 重复图片有多个不同标签
                multi_label_groups += 1
                
                if multi_label_strategy == 'remove_all':
                    # 将重复的标签全部删掉
                    all_removed_groups += 1
                    for img in group:
                        img['processed'] = True
                        img['reason'] = f'removed_multi_label_conflict_{len(labels_in_group)}_labels'
                        self.images_to_remove.append(img)
                    
                    print(f"  多标签组: {len(group)}张图片，标签{list(labels_in_group)}，全部删除")
                    
                elif multi_label_strategy == 'keep_one':
                    # 保留一张（原有行为）
                    if strategy == 'first':
                        keep_img = group[0]
                    elif strategy == 'largest_file':
                        keep_img = max(group, key=lambda x: file_sizes.get(x['relative_path'], 0))
                    elif strategy == 'smallest_file':
                        keep_img = min(group, key=lambda x: file_sizes.get(x['relative_path'], float('inf')))
                    elif strategy == 'highest_resolution':
                        keep_img = max(group, key=lambda x: resolutions.get(x['relative_path'], 0))
                    elif strategy == 'lowest_resolution':
                        keep_img = min(group, key=lambda x: resolutions.get(x['relative_path'], float('inf')))
                    else:
                        keep_img = group[0]
                    
                    keep_img['processed'] = True
                    keep_img['reason'] = f'kept_from_multi_label_group_{len(labels_in_group)}_labels'
                    self.images_to_keep.append(keep_img)
                    
                    # 其余图片标记为删除
                    for img in group:
                        if img != keep_img:
                            img['processed'] = True
                            img['reason'] = f'removed_duplicate_multi_label'
                            self.images_to_remove.append(img)
                    
                    print(f"  多标签组: {len(group)}张图片，标签{list(labels_in_group)}，保留1张")
        
        # 添加未处理的图片到保留列表
        unique_images = 0
        for img in self.image_hashes:
            if not img['processed']:
                img['reason'] = 'unique_image'
                self.images_to_keep.append(img)
                unique_images += 1
        
        print(f"选择完成:")
        print(f"  - 同标签相似组: {same_label_groups}组")
        print(f"  - 多标签相似组: {multi_label_groups}组")
        if multi_label_strategy == 'remove_all':
            print(f"  - 全删除组: {all_removed_groups}组")
        print(f"  - 独特图片: {unique_images}张")
        print(f"  - 最终保留: {len(self.images_to_keep)}张")
        print(f"  - 最终删除: {len(self.images_to_remove)}张")


    def validate_results(self):
        """验证结果的正确性"""
        print("验证结果...")
        
        keep_paths = set(img['relative_path'] for img in self.images_to_keep)
        remove_paths = set(img['relative_path'] for img in self.images_to_remove)
        all_paths = set(img['relative_path'] for img in self.image_hashes)
        
        print(f"调试信息:")
        print(f"  - 总扫描图片数: {len(self.image_hashes)}")
        print(f"  - 所有路径数: {len(all_paths)}")
        print(f"  - 保留路径数: {len(keep_paths)}")
        print(f"  - 删除路径数: {len(remove_paths)}")
        
        # 检查是否有重复扫描的文件
        if len(all_paths) != len(self.image_hashes):
            print(f"警告: 发现重复扫描! 扫描了{len(self.image_hashes)}个文件，但只有{len(all_paths)}个唯一路径")
            
            # 找出重复的路径
            path_counts = {}
            for img in self.image_hashes:
                path = img['relative_path']
                path_counts[path] = path_counts.get(path, 0) + 1
            
            duplicated_paths = {path: count for path, count in path_counts.items() if count > 1}
            if duplicated_paths:
                print("重复扫描的文件:")
                for path, count in duplicated_paths.items():
                    print(f"  - {path}: 扫描了 {count} 次")
        
        # 验证保留和删除列表
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
        
        return True

    def print_results(self, show_details=True, max_groups=10):
        """打印处理结果"""
        print("\n" + "="*60)
        print("处理结果:")
        print(f"- 扫描图片总数: {len(self.image_hashes)}")
        print(f"- 保留图片数: {len(self.images_to_keep)}")
        print(f"- 删除图片数: {len(self.images_to_remove)}")
        print(f"- 相似图片组数: {len(self.similar_groups)}")
        
        if show_details and self.similar_groups:
            print(f"\n相似图片组详情 (显示前{max_groups}组):")
            for i, group in enumerate(self.similar_groups[:max_groups], 1):
                print(f"\n第 {i} 组 ({len(group)} 张图片):")
                for j, img in enumerate(group):
                    status = "保留" if img in self.images_to_keep else "删除"
                    print(f"  {j+1}. [{status}] {img['relative_path']} (标签: {img['label']})")
            
            if len(self.similar_groups) > max_groups:
                print(f"\n... 还有 {len(self.similar_groups) - max_groups} 个相似图片组")

    def save_results(self, output_dir="results"):
        """保存处理结果到CSV文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存要删除的图片列表 - CSV格式
        delete_data = []
        for img in self.images_to_remove:
            delete_data.append({
                'relative_path': img['relative_path'],
                'label': img['label'],
                'hash': img['hash_str']
            })
        
        if delete_data:
            delete_df = pd.DataFrame(delete_data)
            delete_file = output_path / "images_to_delete.csv"
            delete_df.to_csv(delete_file, index=False, encoding='utf-8')
        
        # 保存要保留的图片列表 - CSV格式
        keep_data = []
        for img in self.images_to_keep:
            keep_data.append({
                'relative_path': img['relative_path'],
                'label': img['label'],
                'hash': img['hash_str']
            })
        
        if keep_data:
            keep_df = pd.DataFrame(keep_data)
            keep_file = output_path / "images_to_keep.csv"
            keep_df.to_csv(keep_file, index=False, encoding='utf-8')
        
        # 保存相似图片组详情 - CSV格式
        groups_data = []
        for i, group in enumerate(self.similar_groups, 1):
            for j, img in enumerate(group):
                status = "keep" if img in self.images_to_keep else "remove"
                groups_data.append({
                    'group_id': i,
                    'group_size': len(group),
                    'item_index': j + 1,
                    'status': status,
                    'relative_path': img['relative_path'],
                    'label': img['label'],
                    'hash': img['hash_str']
                })
        
        if groups_data:
            groups_df = pd.DataFrame(groups_data)
            groups_file = output_path / "similar_groups.csv"
            groups_df.to_csv(groups_file, index=False, encoding='utf-8')
        
        print(f"\n结果已保存到 {output_path.absolute()}:")
        if delete_data:
            print(f"- 删除列表: images_to_delete.csv ({len(delete_data)} 条记录)")
        if keep_data:
            print(f"- 保留列表: images_to_keep.csv ({len(keep_data)} 条记录)")
        if groups_data:
            print(f"- 相似组详情: similar_groups.csv ({len(groups_data)} 条记录)")

    def delete_duplicate_images(self, dry_run=True):
        """删除重复图片"""
        if not self.images_to_remove:
            print("没有需要删除的图片")
            return
        
        if dry_run:
            print(f"\n[模拟模式] 将要删除 {len(self.images_to_remove)} 张图片:")
            for img in self.images_to_remove:
                print(f"  - {img['relative_path']}")
            print("\n使用 delete_duplicate_images(dry_run=False) 执行实际删除")
        else:
            print(f"\n开始删除 {len(self.images_to_remove)} 张重复图片...")
            deleted_count = 0
            failed_count = 0
            
            for img in tqdm(self.images_to_remove, desc="删除图片"):
                try:
                    img['path'].unlink()  # 删除文件
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败 {img['relative_path']}: {e}")
                    failed_count += 1
            
            print(f"删除完成: 成功删除 {deleted_count} 张，失败 {failed_count} 张")

    def process(self, hash_type='phash', similarity_threshold=None, selection_strategy='largest_file', 
           multi_label_strategy='remove_all', use_multiprocessing=True, num_workers=None):
        """
        完整的重复图片处理流程（优化版）
        
        Args:
            hash_type: 哈希算法类型 ('phash', 'ahash', 'dhash', 'whash')
            similarity_threshold: 相似度阈值，None表示使用初始化时的值
            selection_strategy: 选择策略 ('first', 'largest_file', 'smallest_file', 'highest_resolution', 'lowest_resolution')
            multi_label_strategy: 多标签处理策略 ('remove_all', 'keep_one')
            use_multiprocessing: 是否使用多进程计算哈希
            num_workers: 工作进程数
        """
        if similarity_threshold is not None:
            self.threshold = similarity_threshold
        
        print(f"\n开始处理重复图片:")
        print(f"  哈希算法: {hash_type}")
        print(f"  相似度阈值: {self.threshold}")
        print(f"  选择策略: {selection_strategy}")
        print(f"  多标签策略: {multi_label_strategy}")
        print(f"  多进程处理: {use_multiprocessing}")
        
        # 1. 扫描并计算哈希（并行优化）
        start_time = time.time()
        self.scan_images(hash_type, use_multiprocessing, num_workers)
        scan_time = time.time() - start_time
        
        if not self.image_hashes:
            print("没有找到可处理的图片")
            return
        
        # 2. 查找相似图片（算法优化）
        start_time = time.time()
        self.find_similar_images_sorted()
        find_time = time.time() - start_time
        
        # 3. 选择保留图片（支持多标签策略）
        start_time = time.time()
        self.select_images_to_keep(selection_strategy, multi_label_strategy)
        select_time = time.time() - start_time
        
        # 4. 验证结果
        if not self.validate_results():
            print("结果验证失败，请检查!")
            return
        
        # 5. 显示结果
        self.print_results()
        
        # 性能报告
        total_time = scan_time + find_time + select_time
        print(f"\n性能报告:")
        print(f"  哈希计算时间: {scan_time:.2f}秒")
        print(f"  相似查找时间: {find_time:.2f}秒")
        print(f"  选择处理时间: {select_time:.2f}秒")
        print(f"  总耗时: {total_time:.2f}秒")
        
        return {
            'total_images': len(self.image_hashes),
            'keep_images': len(self.images_to_keep),
            'remove_images': len(self.images_to_remove),
            'similar_groups': len(self.similar_groups),
            'processing_time': total_time
        }



def main():
    """示例使用"""
    
    # 示例: 处理CSV文件
    cleaner = DuplicateImageCleaner(
        dataset_path="/mnt/7T/xz/wjl/webinat5000_train/train",
        # csv_file="output/web400_cleanlab_index_reserved/images.csv",
        # image_column="image",
        label_column=None,  # 如果没有label列，从路径推断
        hash_size=8,
        threshold=5
    )
    
    results = cleaner.process(
        hash_type='phash',
        selection_strategy='highest_resolution',
        use_multiprocessing=True
    )
    
    # 保存结果
    cleaner.save_results("results_csv")


if __name__ == "__main__":
    main()
