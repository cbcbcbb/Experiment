import os
import re
import argparse
import torch
from multiprocessing import Process, Queue
from propainter.inference2 import Propainter

def find_video_files(root_dir):
    """查找所有符合条件的视频文件"""
    pattern = re.compile(r'final_result.mp4$')
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if pattern.match(fn):
                video_path = os.path.join(dirpath, fn)
                mask_path = os.path.join(dirpath, 'best_mask.mp4')
                output_path = os.path.join(dirpath, 'ppp.mp4')  # 提前构造输出路径
                
                #跳过已有输出文件的目录
                if os.path.exists(output_path):
                    print(f"跳过 {video_path}，输出文件已存在")
                    continue
                
                if os.path.exists(mask_path):
                    video_files.append(video_path)
                else:
                    print(f"跳过 {video_path}，未找到对应mask文件")
    return video_files

def worker(task_queue, gpu_id, config):
    """工作进程函数，处理视频修复任务"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    try:
        # 初始化模型
        propainter = Propainter(config['model_dir'], device=device)
        print(f"GPU {gpu_id}: 模型加载成功")
        
        while True:
            task = task_queue.get()
            if task is None:  # 结束信号
                break
            
            input_video, mask_path, output_path = task
            try:
                #二次检查防止重复处理
                if os.path.exists(output_path):
                    print(f"GPU {gpu_id}: 跳过已存在的 {output_path}")
                    continue
                
                print(f"GPU {gpu_id}: 正在处理 {input_video}")
                propainter.forward(
                    input_video,
                    mask_path,
                    output_path,
                    resize_ratio=1.0,
                    video_length=config['video_length'],
                    ref_stride=config['ref_stride'],
                    neighbor_length=config['neighbor_length'],
                    subvideo_length=config['subvideo_length'],
                    mask_dilation=config['mask_dilation']
                )
                print(f"GPU {gpu_id}: 完成处理 {output_path}")
            except Exception as e:
                print(f"处理失败 {input_video}: {str(e)}")
                
    except Exception as e:
        print(f"GPU {gpu_id} 初始化失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Propainter 批量处理')
    parser.add_argument('--root_dir', default="/data1/cb/rest2M/folder_1/folder_1", help='根目录路径')
    parser.add_argument('--model_dir', default='weights/propainter', help='模型目录')
    parser.add_argument('--video_length', type=int, default=10)
    parser.add_argument('--mask_dilation', type=int, default=8)
    parser.add_argument('--ref_stride', type=int, default=10)
    parser.add_argument('--neighbor_length', type=int, default=10)
    parser.add_argument('--subvideo_length', type=int, default=50)
    args = parser.parse_args()

    config = {
        'model_dir': args.model_dir,
        'video_length': args.video_length,
        'mask_dilation': args.mask_dilation,
        'ref_stride': args.ref_stride,
        'neighbor_length': args.neighbor_length,
        'subvideo_length': args.subvideo_length
    }

    video_files = find_video_files(args.root_dir)
    if not video_files:
        print("未找到需要处理的视频文件")
        return

    task_queue = Queue()
    for video_path in video_files:
        dir_path = os.path.dirname(video_path)
        task = (
            video_path,
            os.path.join(dir_path, 'best_mask.mp4'),
            os.path.join(dir_path, 'ppp.mp4')
        )
        task_queue.put(task)

    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用GPU")

    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker, args=(task_queue, gpu_id, config))
        p.start()
        processes.append(p)
        task_queue.put(None)

    for p in processes:
        p.join()

    print("所有任务处理完成")

if __name__ == '__main__':
    main()
