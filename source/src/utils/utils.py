# 工具函数
import os
import numpy as np


def check_and_create_dir(dir_path):
    """
    检查目录是否存在，若存在且不为空则删除后再创建
    
    参数:
    dir_path: str, 要检查和创建的目录路径
    """
    if os.path.exists(dir_path):
        # 删除目录及其内容
        if os.listdir(dir_path):
            import shutil
            shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def load_clean_indices(save_path):
    """加载清理后的数据索引"""
    if os.path.exists(save_path):
        return np.load(save_path).tolist()
    return None