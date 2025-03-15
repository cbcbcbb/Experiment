import os
import json
import random
import numpy as np
import torch
import cv2
import torch.distributed as dist
from decord import VideoReader, cpu
from einops import rearrange
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import traceback
from diffusers import  AutoencoderKL
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

class VideoMaskDataset(Dataset):

    def __init__(
        self,
        root_dir,
        mask_dir,
        sample_size=512, 
        sample_stride=4,
        sample_n_frames=16,
        mask_dilation_iter=1
    ):
        super().__init__()
        # print(f"Scanning videos from: {root_dir}")
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.mask_dilation_iter = mask_dilation_iter
        self.sample_size=sample_size
        
        # 找出所有含 "video_x.mp4", "best_mask.mp4", "video_x_info.json" 的子文件夹
        self.metadata,self.maskdata = self._scan_videos(root_dir,mask_dir)
        # print(f"Found {len(self.metadata)} valid video folders.")

    def _scan_videos(self,root_dir,mask_dir):
        """
        步骤：
            1) 递归遍历整个 root_dir 和mask_dir
            2) 对这些目录进行检查将其路径信息存入 meta_list
        """
        meta_list = []
        mask_list = []

        # 第一步：遍历所有文件，找到 mask 的目录
        for current_dir, _, files in os.walk(mask_dir):
            for file in files:
                if file.endswith(".png"):
                    mask_list.append(current_dir)
                    break
        
        print(f"共发现 {len(mask_list)} 条符合条件的掩码。")


        for filename in os.listdir(root_dir):  # 遍历当前文件夹
            full_path = os.path.join(root_dir, filename)  # 生成完整路径
            if os.path.isfile(full_path) and filename.endswith(".mp4"):  # 确保是文件且后缀为 .mp4
                meta_list.append({"video_path": full_path})  # 记录符合条件的文件
        
        print(f"共发现 {len(meta_list)} 条符合条件的视频。")

        return meta_list,mask_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        while True:
            try:
                item_dict = self.metadata[idx]
                pixel_values, conditioning_pixel_values, masks = self._get_item_data(item_dict)
                return {
                    "pixel_values": pixel_values, 
                    "conditioning_pixel_values": conditioning_pixel_values,
                    "masks": masks,
                }
            except Exception as e:
                # 如果读视频或其它流程出现问题，则随机换一个 idx 重试
                print(f"Warning: error reading {self.metadata[idx]}: {e}")
                traceback.print_exc()   
                idx = random.randint(0, len(self.metadata) - 1)

    def _get_item_data(self, item_dict):
        """
        读取一个视频文件 + 随机掩码文件，并且做随机抽帧、形态学操作、合成 masked_video。
        """
        video_path = item_dict["video_path"]
 
        # 2. 读取主视频帧 (decord) 
        vr = VideoReader(video_path, ctx=cpu(0))
        video_length = len(vr)

        # 确保采样的 clip 的长度不会超过视频总帧数
        clip_length = (self.sample_n_frames - 1) * self.sample_stride + 1
        if clip_length > video_length:
            clip_length = video_length  # 若视频帧数太少，就取满
        
        start_idx = random.randint(0, video_length - clip_length)
        
        frame_indices = list(range(start_idx, start_idx + clip_length, self.sample_stride))
        if len(frame_indices) < self.sample_n_frames:
            shortage = self.sample_n_frames - len(frame_indices)
            frame_indices.extend([frame_indices[-1]] * shortage)

        # 取到对应帧的数据 shape = (T, H, W, 3)，注意RGB顺序
        frames = vr.get_batch(frame_indices).asnumpy()
        size=frames.shape[1:3]

        #随机读取一个掩码序列进行合成
        selected_folder = random.choice(self.maskdata)
        mask_frames = self._read_mask_video(selected_folder, target_size=size)

        if len(mask_frames) > video_length:
            mask_frames = mask_frames[:video_length]
        else:
            # 计算循环的正向和反向
            forward = mask_frames
            backward = mask_frames[::-1]
            
            # 交替循环，直到达到 video_length
            mask_frames = []
            while len(mask_frames) < video_length:
                mask_frames.extend(forward)
                if len(mask_frames) < video_length:
                    mask_frames.extend(backward)

            # 截取到 video_length 长度
            mask_frames = mask_frames[:video_length]

        # 同样取指定的frame_indices
        selected_mask_frames = [mask_frames[i] for i in frame_indices]

        # 4. 得到 (T, H, W, 3) 的 torch tensor 并归一化到 [0,1]
        # pixel_values = torch.from_numpy(frames).float() / 255.0
        pixel_values = torch.from_numpy(frames).float() 
        pixel_values = rearrange(pixel_values, "t h w c -> t c h w")

        # 掩码只要 1通道 (T, 1, H, W)，且是0或1
        # 这里 selected_mask_frames 已经做过形态学处理，并且是0/255
        # 转成 float => 0.0 或 1.0
        masks = np.stack(selected_mask_frames, axis=0)  # (T, H, W)
        masks = torch.from_numpy(masks).unsqueeze(1).float() 
        # 5. 合成 masked video => (T, 3, H, W)
        #    masked = frame * (1 - mask)
        conditioning_pixel_values = pixel_values * (1 - masks/255)
        # 6. 对 pixel_values, masks, conditioning_pixel_values 做一致的数据增强 / 变换
        #    如随机水平翻转、resize、center crop、归一化(仅对RGB)等。
        #    注意必须对同一个视频的三者做一致操作，否则会对不上。
        # pixel_values, masks, conditioning_pixel_values = \
        #     self._apply_transforms(pixel_values, masks, conditioning_pixel_values)
        
        sample_size = tuple(self.sample_size) if not isinstance(self.sample_size, int) else (self.sample_size, self.sample_size)
        pixel_values, masks,conditioning_pixel_values =self.apply_same_transforms(pixel_values,masks,conditioning_pixel_values,sample_size)
        return pixel_values, conditioning_pixel_values, masks

    def _read_mask_video(self, mask_folder, target_size=None):
        """
        读取指定文件夹中的PNG掩码文件，按文件名顺序处理，并调整尺寸。
        target_size: 目标尺寸元组 (H, W)，若提供则进行resize
        """
        if not os.path.isdir(mask_folder):
            raise IOError(f"Mask folder not found: {mask_folder}")
        
        # 按文件名数字排序PNG文件（假设文件名为0.png, 1.png,...）
        png_files = sorted(
            [f for f in os.listdir(mask_folder) if f.endswith('.png')],
            key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])))) 

        
        mask_frames = []
        for png_file in png_files:
            img_path = os.path.join(mask_folder, png_file)

            # 使用Pillow读取并转换为灰度图
            with Image.open(img_path) as img:
                img = img.convert("L")  # 转为灰度
                frame = np.array(img)
                
            if frame is None:
                raise IOError(f"Could not read mask image: {img_path}")
            
            # 转灰度图
            
            # Resize到目标分辨率（如需）
            if target_size is not None:
                # cv2.resize的尺寸参数是 (宽, 高)，因此用(target_size[1], target_size[0])
                frame = cv2.resize(frame, (target_size[1], target_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # 二值化 + 形态学操作
            m = (frame > 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            m = cv2.erode(m, kernel, iterations=1)
            m = cv2.dilate(m, kernel, iterations=self.mask_dilation_iter)
            
            mask_frames.append(m * 255)  # 转为0-255范围
        
        return mask_frames
  
    def apply_same_transforms(self, video, masks, conditioning, sample_size=(512,512)):
        """
        等效于:
            1) 同步 RandomHorizontalFlip()
            2) Resize(短边=sample_size[0])
            3) 随机Crop(sample_size) —— 同一随机位置，对所有帧一致
        
        参数:
            video        : (T, 3, H, W)
            masks        : (T, 1, H, W)
            conditioning : (T, 3, H, W)
            sample_size  : (H_out, W_out)，比如 (224,224)
        
        返回:
            video_out, masks_out, cond_out 
            分别是 (T,3,H_out,W_out)、(T,1,H_out,W_out)、(T,3,H_out,W_out)
        """
        # -------- 1. 同步随机水平翻转 --------
        do_flip = (random.random() < 0.5)
        if do_flip:
            video = torch.flip(video, dims=[3])
            masks = torch.flip(masks, dims=[3])
            conditioning = torch.flip(conditioning, dims=[3])

        T_, _, _, _ = video.shape
        video_out = []
        masks_out = []
        cond_out  = []

        # -------- 2. 先处理第一帧确定随机crop参数 --------
        # 对第一帧进行Resize，得到统一尺寸（短边=sample_size[0]）
        frame_v0 = video[0]
        frame_v0 = F.resize(frame_v0, sample_size[0])
        _, h, w = frame_v0.shape
        crop_h, crop_w = sample_size

        # 确保裁剪尺寸小于等于resize后的尺寸
        if h >= crop_h and w >= crop_w:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
        else:
            top, left = 0, 0  # 尺寸不足则不裁剪

        # -------- 3. 对每一帧执行Resize -> RandomCrop --------
        for i in range(T_):
            # ~~~ 视频帧处理 ~~~
            frame_v = video[i]  # (3, H, W)
            frame_v = F.resize(frame_v, sample_size[0])
            # 用相同的随机位置进行crop
            frame_v = F.crop(frame_v, top, left, crop_h, crop_w)
            video_out.append(frame_v)

            # ~~~ 条件帧处理 ~~~
            frame_c = conditioning[i]  # (3, H, W)
            frame_c = F.resize(frame_c, sample_size[0])
            frame_c = F.crop(frame_c, top, left, crop_h, crop_w)
            cond_out.append(frame_c)

            # ~~~ 蒙版处理 ~~~
            frame_m = masks[i]  # (1, H, W)
            frame_m = F.resize(frame_m, sample_size[0], interpolation=F.InterpolationMode.NEAREST)
            frame_m = F.crop(frame_m, top, left, crop_h, crop_w)
            masks_out.append(frame_m)

        # 拼接回 (T, C, H_out, W_out)
        video_out = torch.stack(video_out, dim=0)
        cond_out  = torch.stack(cond_out, dim=0)
        masks_out = torch.stack(masks_out, dim=0)

        return  video_out, masks_out, cond_out

def save_batch_as_videos(batch_frames_tensor, out_dir="videos", fps=25):
    """
    将一个批次 (B, T, C, H, W) 的张量保存为多个 mp4 视频文件，每个样本对应一个视频。
    
    参数：
        batch_frames_tensor: 形状 (B, T, C, H, W)，已归一化到 [0,1] 的张量
        out_dir            : 输出文件夹名称
        fps                : 视频帧率
    """
    # 如果在 GPU 上，需要先转到 CPU
    batch_frames_tensor = batch_frames_tensor.detach().cpu()

    # 解析形状
    B, T, C, H, W = batch_frames_tensor.shape

    # 创建输出文件夹
    os.makedirs(out_dir, exist_ok=True)

    # 逐个样本处理
    for b_idx in range(B):
        # 取出第 b_idx 个样本 (T, C, H, W)
        frames_tensor = batch_frames_tensor[b_idx]

        # 目标视频文件路径
        video_path = os.path.join(out_dir, f"video_{b_idx}.mp4")

        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        # 遍历帧
        for t in range(T):
            # (C, H, W) -> (H, W, C)
            frame = frames_tensor[t].numpy().transpose(1, 2, 0).astype(np.uint8)

            # [0,1] -> [0,255]
            # frame = ((frame+1) * 127.5).clip(0,255).astype(np.uint8)

            if C == 3:
                # 如果是 RGB，OpenCV 默认 BGR，需要转换
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif C == 1:
                # 若单通道灰度，简单重复三次做伪彩色写入
                frame = np.repeat(frame, repeats=3, axis=2)
            else:
                raise ValueError(f"不支持的通道数 C={C}（目前只演示 C=3 或 C=1）")

            writer.write(frame)

        writer.release()
        print(f"视频已保存到: {video_path}")
from diffusers.image_processor import VaeImageProcessor

# ------------------- 测试代码 -------------------
if __name__ == "__main__":

    dataset = VideoMaskDataset(
        root_dir="VideoSelection", 
        mask_dir="/data1/cb/separated_masks/separated_masks",
        sample_size=512, 
        sample_stride=1, 
        sample_n_frames=24,
        mask_dilation_iter=3
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    cnt=0
    for batch in dataloader:
    
        conditioning_pixel_values = batch["conditioning_pixel_values"]  # (B, T, 3, H, W)
        
        masks = batch["masks"]                            # (B, T, 1, H, W)
        
        pixel = batch["pixel_values"]
        save_batch_as_videos(conditioning_pixel_values, out_dir=f"Out3/Video{cnt}", fps=4)
        save_batch_as_videos(masks, out_dir=f"Out3/Mask{cnt}", fps=4)
        cnt+=1
        if cnt==5:
            break
        # 这里只跑一个 batch 做演示
 

