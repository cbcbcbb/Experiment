import os
import cv2

# 根目录路径
root_dir = '/data1/cb/rest2M'
# 统计符合条件的视频文件数量
matching_video_count = 0

# 用来存储符合条件的视频文件夹路径
folders_to_delete = []

# 遍历根目录及其子目录
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # 检查文件是否为视频文件（假设以.mp4结尾，可以根据需求修改）
        if filename.endswith('best_mask.mp4'):
            file_path = os.path.join(dirpath, filename)
            # 使用OpenCV读取视频文件
            cap = cv2.VideoCapture(file_path)
            
            # 获取帧数和帧率
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
            cap.release()  # 释放视频流

            # 如果帧数超过300，记录文件夹路径并打印视频信息
            if frame_count > 300:
                matching_video_count += 1
                # folder_path = os.path.dirname(dirpath)  # 获取视频文件所在的文件夹路径
                folders_to_delete.append(dirpath)

                # 打印符合条件的视频信息
                print(f"视频文件：{file_path}")
                print(f"帧数：{frame_count}")
                print(f"帧率：{frame_rate}")
                print(f"对应文件夹：{dirpath}\n")

# 输出符合条件的视频文件总数
print(f"符合条件的视频文件总数：{matching_video_count}")
import shutil
# 确认删除文件夹
if folders_to_delete:
    print("以下文件夹将被删除：")
    for folder in folders_to_delete:
        print(folder)
    
    # 用户确认删除
    confirmation = input("是否确认删除这些文件夹？输入'yes'进行确认，其他输入将取消操作：")
    if confirmation.lower() == 'yes':
        for folder in folders_to_delete:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)  # 删除文件夹及其中的所有文件
                    print(f"已删除文件夹：{folder}")
                except Exception as e:
                    print(f"删除文件夹 {folder} 时发生错误: {e}")
            else:
                print(f"文件夹 {folder} 不存在，跳过删除。")
    else:
        print("操作已取消，不删除文件夹。")
else:
    print("没有符合条件的视频文件夹需要删除。")