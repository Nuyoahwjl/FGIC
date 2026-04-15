import os
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

class BrightnessAnalyzer:
    """图片亮度分析器 - 专门用于检测亮度异常的图片"""
    
    def __init__(self, brightness_threshold=(10, 245)):
        """
        初始化亮度分析器
        
        Args:
            brightness_threshold: 亮度阈值范围 (最小值, 最大值)
        """
        self.brightness_threshold = brightness_threshold
        self.quality_results = []
        self.brightness_issues = []
        
        # 检查numpy是否可用
        self.NUMPY_AVAILABLE = False
        try:
            import numpy as np
            self.NUMPY_AVAILABLE = True
        except ImportError:
            print("警告: NumPy未安装，将使用较慢的纯Python实现")
    
    def analyze_brightness(self, image_path):
        """
        分析单个图片的亮度
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含图片路径、亮度值和其他基本信息的字典
        """
        result = {
            'path': str(image_path),
            'relative_path': None,  # 初始化为None，将在后续处理中设置为正确格式
            'brightness': None,
            'resolution': (0, 0),
            'file_size': 0,
            'error': None
        }
        
        # 尝试从完整路径中提取出目录+文件名格式（如000/hausok.jpg）
        path_obj = Path(image_path)
        parent_name = path_obj.parent.name if path_obj.parent else ''
        
        # 如果父目录是数字格式（如000, 001等），则使用父目录+文件名的格式
        if parent_name and parent_name.isdigit():
            result['relative_path'] = f"{parent_name}/{path_obj.name}"
        else:
            # 否则暂时使用文件名
            result['relative_path'] = path_obj.name
        
        try:
            # 获取文件信息
            result['file_size'] = os.path.getsize(image_path)
            
            with Image.open(image_path) as img:
                # 记录原始分辨率
                width, height = img.size
                result['resolution'] = (width, height)
                
                # 计算是否需要采样 (对于大图)
                max_pixels = 1024 * 1024  # 100万像素作为阈值
                total_pixels = width * height
                
                # 直接转换为灰度图以提高效率
                img_gray = img.convert('L')
                
                if self.NUMPY_AVAILABLE:
                    # 使用numpy的高效实现
                    # 对于大图进行降采样
                    if total_pixels > max_pixels:
                        # 计算缩放比例
                        scale_factor = (max_pixels / total_pixels) ** 0.5
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        # 缩小图片以加快处理
                        img_gray = img_gray.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 转换为numpy数组
                    img_array = np.array(img_gray)
                    # 直接计算平均亮度
                    result['brightness'] = np.mean(img_array)
                else:
                    # 不使用numpy的纯Python实现
                    # 对于大图进行采样
                    if total_pixels > max_pixels:
                        # 计算采样步长
                        step = max(1, int((total_pixels / max_pixels) ** 0.5))
                        total_brightness = 0
                        pixel_count = 0
                        
                        # 采样读取像素
                        for x in range(0, width, step):
                            for y in range(0, height, step):
                                total_brightness += img_gray.getpixel((x, y))
                                pixel_count += 1
                        
                        result['brightness'] = total_brightness / pixel_count if pixel_count > 0 else 0
                    else:
                        # 对于小图，读取整个图像数据
                        pixels = list(img_gray.getdata())
                        result['brightness'] = sum(pixels) / len(pixels)
        
        except Exception as e:
            result['error'] = str(e)
            print(f"处理图片 {image_path} 时出错: {e}")
        
        return result
    
    def prefilter_images(self, image_paths, min_file_size=1024):
        """
        预过滤图片列表，快速跳过明显不需要完整处理的图片
        
        Args:
            image_paths: 图片路径列表
            min_file_size: 最小文件大小阈值（字节）
            
        Returns:
            list: 过滤后的图片路径列表
        """
        print(f"预过滤 {len(image_paths)} 张图片...")
        
        filtered_paths = []
        skipped_count = 0
        
        for path in image_paths:
            try:
                # 快速检查文件是否存在
                if not os.path.exists(path):
                    skipped_count += 1
                    continue
                
                # 快速检查文件大小 - 特别小的文件可能是无效图片
                file_size = os.path.getsize(path)
                if file_size < min_file_size:
                    skipped_count += 1
                    continue
                
                filtered_paths.append(path)
            except Exception:
                # 任何异常都跳过此文件
                skipped_count += 1
                continue
        
        print(f"预过滤完成：保留 {len(filtered_paths)} 张图片，跳过 {skipped_count} 张图片")
        return filtered_paths
    
    def analyze_brightness_batch(self, image_paths, use_multiprocessing=True, num_workers=None, chunk_size=100, use_prefilter=True):
        """
        批量分析图片亮度
        
        Args:
            image_paths: 图片路径列表
            use_multiprocessing: 是否使用多进程
            num_workers: 工作进程数
            chunk_size: 任务分块大小，用于减少进程间通信开销
            use_prefilter: 是否使用预过滤机制
        """
        print(f"开始分析 {len(image_paths)} 张图片的亮度...")
        
        # 预过滤图片
        if use_prefilter:
            image_paths = self.prefilter_images(image_paths)
            if not image_paths:
                print("没有需要分析的有效图片")
                self.quality_results = []
                return self.quality_results
        
        # 动态调整是否使用多进程 - 小批量任务不使用多进程以避免额外开销
        should_use_multiprocessing = use_multiprocessing and len(image_paths) > 5
        
        if should_use_multiprocessing:
            # 优化进程数计算 - 对于大量小文件可以使用更多进程
            cpu_count = mp.cpu_count()
            # 根据任务数量和CPU核心数动态调整进程数
            if num_workers is None:
                if len(image_paths) < cpu_count * 10:
                    # 少量图片，使用较少进程
                    num_workers = min(max(1, cpu_count // 2), len(image_paths))
                else:
                    # 大量图片，充分利用CPU
                    num_workers = min(cpu_count, len(image_paths))
                    # 对于特别多的任务，可以适当增加进程数以提高IO利用率
                    if len(image_paths) > cpu_count * 100:
                        num_workers = min(cpu_count * 2, len(image_paths))
            
            print(f"使用 {num_workers} 个进程并行分析...")
            
            # 优化任务分块大小
            optimal_chunk_size = max(1, chunk_size)
            
            with mp.Pool(processes=num_workers) as pool:
                # 使用imap_unordered提高处理效率，尤其是当任务执行时间不均时
                # 增加chunksize参数以减少进程间通信开销
                self.quality_results = list(tqdm(
                    pool.imap_unordered(self.analyze_brightness, image_paths, chunksize=optimal_chunk_size),
                    total=len(image_paths),
                    desc="分析亮度"
                ))
        else:
            # 单进程处理 - 使用列表推导式优化
            self.quality_results = [self.analyze_brightness(path) for path in tqdm(image_paths, desc="分析亮度")]
        
        print(f"亮度分析完成，共分析 {len(self.quality_results)} 张图片")
        return self.quality_results
    
    def filter_brightness_issues(self):
        """
        筛选亮度异常的图片
        
        Returns:
            list: 亮度异常的图片列表
        """
        print(f"筛选亮度异常的图片，阈值范围: {self.brightness_threshold}...")
        
        self.brightness_issues = []
        min_bright, max_bright = self.brightness_threshold
        
        for result in self.quality_results:
            if result['brightness'] is not None:
                brightness = result['brightness']
                issues = []
                
                # 检查亮度是否异常
                if brightness < min_bright:
                    issues.append(f'过暗(亮度:{brightness:.1f}<{min_bright})')
                elif brightness > max_bright:
                    issues.append(f'过亮(亮度:{brightness:.1f}>{max_bright})')
                
                if issues:
                    result['quality_issues'] = '; '.join(issues)
                    result['issue_count'] = len(issues)
                    self.brightness_issues.append(result)
        
        # 按亮度值排序（先过暗，再过亮）
        self.brightness_issues.sort(key=lambda x: (
            0 if '过暗' in x['quality_issues'] else 1,  # 先显示过暗的图片
            x['brightness'] if '过暗' in x['quality_issues'] else -x['brightness']  # 过暗按亮度升序，过亮按亮度降序
        ))
        
        print(f"发现 {len(self.brightness_issues)} 张亮度异常的图片")
        return self.brightness_issues
    
    def save_brightness_results(self, output_dir="./results"):
        """
        保存亮度分析结果，输出两个CSV文件：一个亮度正常，一个亮度异常
        
        Args:
            output_dir: 输出目录
        
        Returns:
            tuple: (异常图片CSV路径, 正常图片CSV路径)
        """
        # 使用os.makedirs一次创建目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建输出文件路径
        issues_csv_path = os.path.join(output_dir, "brightness_issues.csv")
        normal_csv_path = os.path.join(output_dir, "brightness_normal.csv")
        
        # 准备亮度异常图片的数据
        issues_rows = []
        for img in self.brightness_issues:
            # 预计算所有字段值
            resolution_str = f"{img['resolution'][0]}x{img['resolution'][1]}" if img['resolution'] else "N/A"
            file_size_kb = img['file_size'] / 1024.0
            
            issues_rows.append({
                'relative_path': img.get('relative_path', 'N/A'),
                'issue_count': img.get('issue_count', 0),
                'quality_issues': img.get('quality_issues', ''),
                'resolution': resolution_str,
                'file_size_kb': file_size_kb,
                'brightness': img.get('brightness', '')
            })
        
        # 准备亮度正常图片的数据
        # 创建异常图片路径集合，用于快速查找
        issue_paths = {img['path'] for img in self.brightness_issues}
        
        normal_rows = []
        for img in self.quality_results:
            # 跳过异常图片和有错误的图片
            if img['path'] not in issue_paths and img['brightness'] is not None and not img['error']:
                resolution_str = f"{img['resolution'][0]}x{img['resolution'][1]}" if img['resolution'] else "N/A"
                file_size_kb = img['file_size'] / 1024.0
                
                normal_rows.append({
                    'relative_path': img.get('relative_path', 'N/A'),
                    'resolution': resolution_str,
                    'file_size_kb': file_size_kb,
                    'brightness': img.get('brightness', '')
                })
        
        # 批量写入异常图片CSV
        if issues_rows:
            with open(issues_csv_path, 'w', newline='', encoding='utf-8', buffering=8192) as f:
                fieldnames = ['relative_path', 'issue_count', 'quality_issues', 'resolution', 
                             'file_size_kb', 'brightness']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(issues_rows)  # 批量写入比逐行写入更高效
            print(f"亮度异常图片列表已保存至: {issues_csv_path}，共 {len(issues_rows)} 张")
        else:
            print("没有亮度异常的图片需要保存")
        
        # 批量写入正常图片CSV
        if normal_rows:
            with open(normal_csv_path, 'w', newline='', encoding='utf-8', buffering=8192) as f:
                fieldnames = ['relative_path', 'resolution', 'file_size_kb', 'brightness']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(normal_rows)
            print(f"亮度正常图片列表已保存至: {normal_csv_path}，共 {len(normal_rows)} 张")
        else:
            print("没有亮度正常的图片需要保存")
        
        return issues_csv_path, normal_csv_path
        
    def filter_from_csv(self, csv_file, base_dir=".", output_dir="./results", use_prefilter=True):
        """
        从CSV文件中筛选亮度异常的图片，并输出两个CSV文件
        
        Args:
            csv_file: 包含图片列表的CSV文件路径
            base_dir: 图片文件的基础目录，用于构建完整路径
            output_dir: 输出目录
            use_prefilter: 是否使用预过滤机制
            
        Returns:
            tuple: (异常图片CSV路径, 正常图片CSV路径)
        """
        print(f"从CSV文件 {csv_file} 中筛选亮度异常的图片...")
        
        # 读取CSV文件中的图片列表
        image_paths = []
        relative_paths = []
        csv_data = []
        
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'image' in row:
                    rel_path = row['image']
                    # 构建完整路径
                    full_path = os.path.join(base_dir, rel_path)
                    # 确保路径使用正确的分隔符
                    full_path = str(Path(full_path))
                    
                    image_paths.append(full_path)
                    relative_paths.append(rel_path)
                    csv_data.append(row)
        
        print(f"已读取 {len(image_paths)} 张图片的信息")
        
        # 批量分析亮度
        self.analyze_brightness_batch(image_paths, use_multiprocessing=True, use_prefilter=use_prefilter)
        
        # 筛选亮度异常的图片
        self.filter_brightness_issues()
        
        # 创建相对路径到分析结果的映射
        rel_path_to_result = {}
        for i, result in enumerate(self.quality_results):
            # 用CSV中的relative_path替换结果中的
            if i < len(relative_paths):
                result['relative_path'] = relative_paths[i]
            rel_path_to_result[result['relative_path']] = result
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 输出两个CSV文件
        issue_csv_path = os.path.join(output_dir, "brightness_issues_from_csv.csv")
        normal_csv_path = os.path.join(output_dir, "brightness_normal_from_csv.csv")
        
        # 预计算异常图片路径集合，用于快速查找
        issue_paths_set = {img['relative_path'] for img in self.brightness_issues}
        
        # 创建异常图片字典，用于快速查找异常信息
        issue_info_dict = {img['relative_path']: img.get('quality_issues', '') for img in self.brightness_issues}
        
        # 准备字段名
        if csv_data:
            issue_fieldnames = list(csv_data[0].keys()) + ['brightness', 'quality_issues']
            normal_fieldnames = list(csv_data[0].keys()) + ['brightness']
        else:
            issue_fieldnames = ['relative_path', 'brightness', 'quality_issues']
            normal_fieldnames = ['relative_path', 'brightness']
        
        # 预分类数据，减少写入时的操作
        issue_rows = []
        normal_rows = []
        
        for row in csv_data:
            rel_path = row['image']
            if rel_path in rel_path_to_result:
                result = rel_path_to_result[rel_path]
                # 创建新行而不是复制整行，减少内存使用
                new_row = {k: v for k, v in row.items()}  # 只复制需要的字段
                new_row['brightness'] = result.get('brightness', '')
                
                # 使用集合快速判断是否为异常图片，避免多次循环
                if rel_path in issue_paths_set:
                    # 异常图片
                    new_row['quality_issues'] = issue_info_dict.get(rel_path, '')
                    issue_rows.append(new_row)
                else:
                    # 正常图片
                    normal_rows.append(new_row)
        
        # 批量写入异常图片CSV
        if issue_rows:
            with open(issue_csv_path, 'w', newline='', encoding='utf-8', buffering=8192) as issue_file:
                issue_writer = csv.DictWriter(issue_file, fieldnames=issue_fieldnames)
                issue_writer.writeheader()
                issue_writer.writerows(issue_rows)
        
        # 批量写入正常图片CSV
        if normal_rows:
            with open(normal_csv_path, 'w', newline='', encoding='utf-8', buffering=8192) as normal_file:
                normal_writer = csv.DictWriter(normal_file, fieldnames=normal_fieldnames)
                normal_writer.writeheader()
                normal_writer.writerows(normal_rows)
        
        issue_count = len(issue_rows)
        normal_count = len(normal_rows)
        
        print(f"亮度异常图片已保存至: {issue_csv_path}，共 {issue_count} 张")
        print(f"亮度正常图片已保存至: {normal_csv_path}，共 {normal_count} 张")
        
        return issue_csv_path, normal_csv_path
    
    def print_brightness_report(self, max_display=20):
        """
        打印亮度分析报告
        
        Args:
            max_display: 最大显示数量
        """
        print(f"\n=== 亮度异常图片报告 ===")
        print(f"总计发现 {len(self.brightness_issues)} 张亮度异常的图片")
        
        # 统计过暗和过亮的数量
        too_dark_count = sum(1 for img in self.brightness_issues if '过暗' in img['quality_issues'])
        too_bright_count = sum(1 for img in self.brightness_issues if '过亮' in img['quality_issues'])
        
        print(f"过暗图片: {too_dark_count} 张")
        print(f"过亮图片: {too_bright_count} 张")
        
        # 显示部分异常图片
        print(f"\n前 {min(max_display, len(self.brightness_issues))} 张异常图片详情:")
        for i, img in enumerate(self.brightness_issues[:max_display]):
            print(f"{i+1:3d}. {img['relative_path']}")
            print(f"   问题: {img['quality_issues']}")
            print(f"   分辨率: {img['resolution'][0]}x{img['resolution'][1]}")
            print(f"   文件大小: {img['file_size']//1024:.1f}KB")
            print(f"   亮度: {img['brightness']:.1f}")


def find_images(directory):
    """
    查找目录下所有的图片文件
    
    Args:
        directory: 要搜索的目录
        
    Returns:
        list: 图片路径列表
    """
    # 使用fnmatch代替多次endswith检查，提高效率
    import fnmatch
    
    # 预编译所有扩展名模式，避免重复字符串操作
    image_patterns = {
        '*.jpg', '*.jpeg', '*.png', '*.bmp', 
        '*.gif', '*.tiff', '*.webp',
        '*.JPG', '*.JPEG', '*.PNG', '*.BMP',
        '*.GIF', '*.TIFF', '*.WEBP'
    }
    
    image_paths = []
    
    # 使用os.scandir替代os.walk，提供更好的性能
    def scan_directory(path):
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        # 递归扫描子目录
                        scan_directory(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        # 使用fnmatch检查文件扩展名
                        for pattern in image_patterns:
                            if fnmatch.fnmatch(entry.name, pattern):
                                # 使用Path对象确保跨平台路径一致性
                                image_paths.append(str(Path(entry.path)))
                                break
        except (PermissionError, FileNotFoundError):
            # 忽略无权限或不存在的目录
            pass
    
    # 开始扫描
    scan_directory(directory)
    
    return image_paths

def main():
    """
    主函数，支持两种模式：
    1. CSV模式：从CSV文件读取图片路径进行分析
    2. 目录模式：直接分析指定目录下的图片
    
    支持通过命令行参数切换模式
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='图片亮度异常检测工具')
    
    # 添加模式选择参数
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--csv', action='store_true', help='使用CSV模式，从CSV文件读取图片路径')
    mode_group.add_argument('--dir', action='store_true', help='使用目录模式，直接分析目录下的图片')
    
    # 添加其他参数
    parser.add_argument('--csv-file', default="./output/web5000_cleanlab_index/indices/clean_image_paths.csv", 
                       help='CSV文件路径 (默认: ./results_csv/images_to_keep.csv)')
    parser.add_argument('--dir-path', default="/mnt/7T/xz/wjl/webinat5000_train/train", 
                       help='要分析的目录路径 (默认: ./test)')
    parser.add_argument('--base-dir', default="/mnt/7T/xz/wjl/webinat5000_train/train", 
                       help='CSV模式下图片的基础目录 (默认自动检测)')
    parser.add_argument('--output-dir', default="./web5000_indices", 
                       help='结果输出目录 (默认: ./results')
    parser.add_argument('--min-bright', type=int, default=10, 
                       help='最小亮度阈值 (默认: 10)')
    parser.add_argument('--max-bright', type=int, default=245, 
                       help='最大亮度阈值 (默认: 245)')
    parser.add_argument('--no-multiprocessing', action='store_true', 
                       help='禁用多进程处理')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置亮度阈值
    brightness_threshold = (args.min_bright, args.max_bright)
    
    # 使用Path对象处理所有路径，确保跨平台一致性
    output_dir = Path(args.output_dir).resolve()
    
    # 创建输出目录 - 使用parents=True确保所有父目录都被创建
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定使用哪种模式
    # 如果没有明确指定模式，自动检测：优先使用CSV模式（如果文件存在）
    use_csv_mode = args.csv
    use_dir_mode = args.dir
    
    if not use_csv_mode and not use_dir_mode:
        # 自动检测模式
        csv_path = Path(args.csv_file).resolve()
        use_csv_mode = csv_path.exists()
        use_dir_mode = not use_csv_mode
    
    # 执行CSV模式
    if use_csv_mode:
        csv_path = Path(args.csv_file).resolve()
        if not csv_path.exists():
            print(f"错误：CSV文件不存在: {csv_path}")
            print("请使用 --dir 参数切换到目录模式")
            return
            
        print(f"\n=== 从CSV文件筛选亮度异常图片 ({csv_path}) ===")
        analyzer = BrightnessAnalyzer(brightness_threshold=brightness_threshold)
        
        # 确定图片的基础目录
        if args.base_dir:
            base_dir = str(Path(args.base_dir).resolve())
        else:
            # 自动检测基础目录
            possible_base_dirs = [Path('./train').resolve()]
            base_dir = str(Path('.').resolve())
            
            for bd in possible_base_dirs:
                if bd.exists() and bd.is_dir():
                    # 检查是否有匹配的文件
                    if any((bd / p).exists() for p in ["000", "001"]):
                        base_dir = str(bd)
                        break
        
        print(f"使用基础目录: {base_dir}")
        
        # 从CSV文件筛选并输出结果
        analyzer.filter_from_csv(str(csv_path), base_dir=base_dir, output_dir=str(output_dir))
    
    # 执行目录模式
    elif use_dir_mode:
        dir_path = Path(args.dir_path).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"错误：目录不存在或不是有效目录: {dir_path}")
            return
            
        print(f"\n=== 直接分析目录中的图片 ({dir_path}) ===")
        
        # 查找所有图片
        print(f"正在查找 {dir_path} 目录下的图片...")
        image_paths = find_images(str(dir_path))
        print(f"找到 {len(image_paths)} 张图片")
        
        if not image_paths:
            print("没有找到图片文件，程序退出")
            return
        
        # 创建亮度分析器
        analyzer = BrightnessAnalyzer(brightness_threshold=brightness_threshold)
        
        # 批量分析亮度
        analyzer.analyze_brightness_batch(image_paths, use_multiprocessing=not args.no_multiprocessing)
        
        # 筛选亮度异常的图片
        analyzer.filter_brightness_issues()
        
        # 保存结果
        issues_csv, normal_csv = analyzer.save_brightness_results(output_dir=str(output_dir))
        
        # 打印报告
        analyzer.print_brightness_report()

    print("\n分析完成！")

if __name__ == "__main__":
    main()