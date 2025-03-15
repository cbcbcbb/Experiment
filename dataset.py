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
    """
    从指定的 root_dir 下，扫描所有含有视频文件、对应 best_mask.mp4 掩码文件、
    以及 info.json（内含文本）的子文件夹，并且在 __getitem__ 中返回:
        "pixel_values"：原始视频的采样片段
        "conditioning_pixel_values"：被掩码（挖空）后的采样片段
        "masks"：处理后的掩码片段
        "text"：JSON中读取到的 caption
    """
    def __init__(
        self,
        root_dir,
        sample_size=512, 
        sample_stride=4,
        sample_n_frames=16,
        mask_dilation_iter=1
    ):
        super().__init__()
        # print(f"Scanning videos from: {root_dir}")
        self.root_dir = root_dir
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.mask_dilation_iter = mask_dilation_iter
        self.sample_size=sample_size
        
        # 找出所有含 "video_x.mp4", "best_mask.mp4", "video_x_info.json" 的子文件夹
        self.metadata = self._scan_videos(root_dir)
        # print(f"Found {len(self.metadata)} valid video folders.")

        # 定义要将视频帧转换为 tensor 的图像变换
        # 注意：我们要对 video/mask 做一致的变换，因此需要小心处理 (见后续的 _apply_transforms 函数)
        self._transform_fn = T.Compose([
            # 这里的随机水平翻转，如果视频是多帧，需要保证所有帧同步翻转
            # 因此我们后面会手写一个 “对整个 video tensor 同时做flip” 的逻辑。
            # 如果你希望使用 torchvision.RandomHorizontalFlip，需要给每一帧/掩码共用一个随机状态。
            #
            # 所以这里仅演示 Resize + CenterCrop + Normalize。随机flip在后面单独演示。
            
            T.Resize(sample_size),  # 等比或直接指定 (sample_size, sample_size)
            T.CenterCrop((sample_size, sample_size)),
            # 最后做 [-1,1] 的归一化
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _scan_videos(self,root_dir):
        """
        步骤：
            1) 递归遍历整个 root_dir，收集所有包含 ppp.mp4 的子目录；
            2) 对这些目录进行检查，若 video_*.mp4、best_mask.mp4、*_info.json、ppp.mp4 都齐全，则将其路径信息存入 meta_list；
        """
        meta_list = []
        directories_with_ppp = set()

        # 第一步：遍历所有文件，找到 ppp.mp4 的目录
        for current_dir, _, files in os.walk(root_dir):
            if "ppp.mp4" in files:
                directories_with_ppp.add(current_dir)

        # 第二步：对收集到的目录进行文件检查
        for dir_with_ppp in directories_with_ppp:
            files_in_dir = os.listdir(dir_with_ppp)

            video_path = None
            mask_path  = None
            info_path  = None
            ppp_path   = None

            for filename in files_in_dir:
                full_path = os.path.join(dir_with_ppp, filename)

                if filename == "best_mask.mp4":
                    mask_path = full_path
                elif filename == "ppp.mp4":
                    ppp_path = full_path
                elif filename.endswith("_info.json"):
                    info_path = full_path
                elif filename.startswith("video_") and filename.endswith(".mp4"):
                    # 如果可能存在多个 video_ 文件，则可改为存储到列表
                    # 此处只记录第一个，视需求自行调整
                    video_path = full_path

            # 若所需文件都存在，则保存
            if video_path and mask_path and info_path and ppp_path:
                meta_list.append({
                    "video_path": video_path,
                    "mask_path":  mask_path,
                    "info_path":  info_path,
                    "ppp_path":   ppp_path,
                })

        print(f"共发现 {len(meta_list)} 条符合条件的记录。")
        return meta_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        while True:
            try:
                item_dict = self.metadata[idx]
                pixel_values, conditioning_pixel_values, masks, ppp_values, text = self._get_item_data(item_dict)
                return {
                    "pixel_values": pixel_values, 
                    "conditioning_pixel_values": conditioning_pixel_values,
                    "masks": masks,
                    "ppp_values" : ppp_values,
                    "text": text
                }
            except Exception as e:
                # 如果读视频或其它流程出现问题，则随机换一个 idx 重试
                print(f"Warning: error reading {self.metadata[idx]}: {e}")
                traceback.print_exc()   
                idx = random.randint(0, len(self.metadata) - 1)

    def _get_item_data(self, item_dict):
        """
        读取一个视频文件 + 掩码文件，并且做随机抽帧、形态学操作、合成 masked_video。
        """
        video_path = item_dict["video_path"]
        mask_path  = item_dict["mask_path"]
        info_path  = item_dict["info_path"]
        ppp_path   = item_dict["ppp_path"]
        # 1. 读取文本
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
        text = info_data.get("caption", "")
        # 2. 读取主视频帧 (decord) 和 Propainter处理后的帧
        vr = VideoReader(video_path, ctx=cpu(0))
        video_length = len(vr)
        ppp = VideoReader(ppp_path, ctx=cpu(0))
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
        ppp_frames = ppp.get_batch(frame_indices).asnumpy()
        # 3. 读取掩码视频 & 形态学处理
        #    这里演示一次性把整段掩码读到内存，然后再去索引对应 frames，
        #    也可以用和上面同样 decord.get_batch 逻辑，但这里演示和你的cv2伪码更相似。
        mask_frames = self._read_mask_video(mask_path)
        # 如果掩码视频帧数比原视频少或多，这里需要加一个最简的保护
        # 先把 mask_frames 修剪到和 video_length 一样多
        if len(mask_frames) > video_length:
            mask_frames = mask_frames[:video_length]
        
        # 同样取指定的frame_indices
        selected_mask_frames = [mask_frames[i] for i in frame_indices]

        # 4. 得到 (T, H, W, 3) 的 torch tensor 并归一化到 [0,1]
        # pixel_values = torch.from_numpy(frames).float() / 255.0
        pixel_values = torch.from_numpy(frames).float() 
        pixel_values = rearrange(pixel_values, "t h w c -> t c h w")
        # ppp_values=torch.from_numpy(ppp_frames).float() / 255.0
        ppp_values=torch.from_numpy(ppp_frames).float() 
        ppp_values = rearrange(ppp_values, "t h w c -> t c h w")

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
        ppp_values, pixel_values, masks,conditioning_pixel_values =self.apply_same_transforms(ppp_values, pixel_values,masks,conditioning_pixel_values,sample_size)
        return pixel_values, conditioning_pixel_values, masks, ppp_values,text

    def _read_mask_video(self, mask_path):
        """
        使用 cv2 把整个视频读进来，并做形态学处理(先 resize 再腐蚀+膨胀)。
        注意：如果你希望和原视频同分辨率，可以先用 decord 取原视频的 H,W 再在这里 resize。
        这里仅是一个示例。
        """
        cap = cv2.VideoCapture(mask_path)
        if not cap.isOpened():
            raise IOError(f"Could not open mask video: {mask_path}")
        
        mask_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame shape = (H, W, 3)，OpenCV是BGR，需要转成灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 这里不清楚你的视频原始分辨率和后面如何匹配，这里直接不做resize，
            # 仅演示怎么形态学处理。如果你要resize跟视频同样大小，可以等在
            # _apply_transforms() 里做统一resize；或者这里也可以做一次。
            
            # gray = cv2.resize(gray, (someW, someH), interpolation=cv2.INTER_NEAREST)

            # m > 0 => 1，否则0
            m = (gray > 0).astype(np.uint8)
            # 腐蚀1次
            m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            # 膨胀 mask_dilation_iter 次
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=self.mask_dilation_iter)

            # 存到列表
            mask_frames.append(m * 255)

        cap.release()
        return mask_frames

    
    def apply_same_transforms(self, ppp_video, video, masks, conditioning, sample_size=(512,512)):
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
            ppp_video = torch.flip(ppp_video, dims=[3])
            video = torch.flip(video, dims=[3])
            masks = torch.flip(masks, dims=[3])
            conditioning = torch.flip(conditioning, dims=[3])

        T_, _, _, _ = video.shape
        ppp_video_out = []
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

            # ~~~ ppp帧处理 ~~~
            frame_p = ppp_video[i]  # (3, H, W)
            frame_p = F.resize(frame_p, sample_size[0])
            frame_p = F.crop(frame_p, top, left, crop_h, crop_w)
            ppp_video_out.append(frame_p)

            # ~~~ 蒙版处理 ~~~
            frame_m = masks[i]  # (1, H, W)
            frame_m = F.resize(frame_m, sample_size[0], interpolation=F.InterpolationMode.NEAREST)
            frame_m = F.crop(frame_m, top, left, crop_h, crop_w)
            masks_out.append(frame_m)

        # 拼接回 (T, C, H_out, W_out)
        video_out = torch.stack(video_out, dim=0)
        cond_out  = torch.stack(cond_out, dim=0)
        masks_out = torch.stack(masks_out, dim=0)
        ppp_video_out= torch.stack(ppp_video_out, dim=0)

        return ppp_video_out, video_out, masks_out, cond_out

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
        root_dir="/data1/cb/rest2M", 
        sample_size=512, 
        sample_stride=1, 
        sample_n_frames=22,
        mask_dilation_iter=3
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    # vae = AutoencoderKL.from_pretrained(
    #     "weights/sd-vae-ft-mse"
    # ).to('cuda:0')

    for batch in dataloader:
    
        # #prepare images
        # ppp_values = batch["ppp_values"].to('cuda:0')
        # masks = batch["masks"].to('cuda:0')
        # conditioning_pixel_values=batch["conditioning_pixel_values"].to('cuda:0')
        # for b in range(ppp_values.shape[0]):  # 遍历 batch 维度
        #     for t in range(ppp_values.shape[1]):  # 遍历时间维度（帧数）
        #         # 取出当前批次 `b` 的第 `t` 帧图像
        #         frame = ppp_values[b, t]/255  # 形状 (3, H, W)
        #         mask = masks[b, t]/255  # 形状 (1, H, W)
        #         masked_frame = conditioning_pixel_values[b, t]/255
        #         ppp_values[b, t]=(frame-0.5)/0.5
        #         masks[b, t]=(mask-0.5)/0.5
        #         conditioning_pixel_values[b, t]=(masked_frame-0.5)/0.5
               
        # print(ppp_values.shape,ppp_values.min(),ppp_values.max())

        # with torch.no_grad():
        #     #prepare latents
        #     video_length = ppp_values.shape[1]
        #     ppp_values = rearrange(ppp_values, "b f c h w -> (b f) c h w")
        #     latents = vae.encode(ppp_values).latent_dist.sample() * vae.config.scaling_factor

        #     #prepare condition latent
        #     video_length = conditioning_pixel_values.shape[1]
        #     conditioning_pixel_values = rearrange(conditioning_pixel_values, "b f c h w -> (b f) c h w")
        #     conditioning_latents = vae.encode(conditioning_pixel_values).latent_dist.sample() * vae.config.scaling_factor
        #     conditioning_latents = rearrange(conditioning_latents, "(b f) c h w -> b f c h w", f=video_length)

        # masks = rearrange(masks, "b f c h w -> (b f) c h w")
        # #prepare mask
        # masks = torch.nn.functional.interpolate(
        #     masks, 
        #     size=(
        #         latents.shape[-2], 
        #         latents.shape[-1]
        #     )
        # )
        # masks = rearrange(masks, "(b f) c h w -> b f c h w",b=conditioning_latents.shape[0], f=conditioning_latents.shape[1])
        # conditioning_latents=torch.concat([conditioning_latents,masks],2)
        # conditioning_latents=rearrange(conditioning_latents, "b f c h w -> (b f) c h w")



        # with torch.no_grad():
        #     #############################################
        #     # 解码主潜变量 latents -> ppp_values 对应的视频
        #     #############################################
        #     # 计算 batch_size 和 video_length
        #     batch_size = batch["masks"].shape[0]
        #     video_length =  batch["masks"].shape[1]
        #     print(batch_size,video_length)
        #     # 反缩放 + 解码
        #     decoded_latents = latents / vae.config.scaling_factor
        #     pixel_values = vae.decode(decoded_latents).sample  # shape: (b*f, 3, H, W)

        #     # 转换到 [0, 255] 范围并重组形状
        #     pixel_values = (pixel_values * 0.5 + 0.5).clamp(0, 1)  #  VAE 输出范围是 [-1, 1]
        #     pixel_values = rearrange(pixel_values, "(b f) c h w -> b f c h w", b=batch_size, f=video_length)
        #     pixel_values = (pixel_values * 255).to(torch.uint8)  # shape: (B, F, 3, H, W)

        #     #############################################
        #     # 解码条件潜变量 conditioning_latents 对应的视频
        #     #############################################
        #     # 注意：需要从拼接后的张量中分离原始条件潜变量（假设原始通道数为4）
        #     conditioning_latents_clean = conditioning_latents[:, :4, :, :]  # 分割前4个通道（假设原始条件潜变量是4通道）
        #     conditioning_latents_clean = conditioning_latents_clean / vae.config.scaling_factor
            
        #     # 解码条件潜变量
        #     conditioning_pixel_values = vae.decode(conditioning_latents_clean).sample
        #     conditioning_pixel_values = (conditioning_pixel_values * 0.5 + 0.5).clamp(0, 1)
        #     conditioning_pixel_values = rearrange(conditioning_pixel_values, "(b f) c h w -> b f c h w", b=batch_size, f=video_length)
        #     conditioning_pixel_values = (conditioning_pixel_values * 255).to(torch.uint8)

        #     #############################################
        #     # 处理掩码 masks 为视频格式
        #     #############################################
        #     # 反插值到原始分辨率（假设原始分辨率已知）
        #     original_h = batch["masks"].shape[-2]
        #     original_w = batch["masks"].shape[-1]
        #     masks = torch.nn.functional.interpolate(
        #         masks.to(torch.float32), 
        #         size=(1,original_h, original_w),
        #         mode="nearest"
        #     )
        #     masks = (masks * 255).to(torch.uint8)  # 转换为 0-255 范围


        # save_batch_as_videos(pixel_values, out_dir="Out/Videos", fps=4)
        # save_batch_as_videos(conditioning_pixel_values, out_dir="Out/MaskedVideos", fps=4)
        # save_batch_as_videos(masks, out_dir="Out/Masks", fps=4)
        
        conditioning_pixel_values = batch["conditioning_pixel_values"]  # (B, T, 3, H, W)
        
        masks = batch["masks"]                            # (B, T, 1, H, W)
        
        ppp = batch["ppp_values"]
        save_batch_as_videos(ppp, out_dir="Out/Ori/Videos", fps=4)
        save_batch_as_videos(conditioning_pixel_values, out_dir="Out/Ori/MaskedVideos", fps=4)
        save_batch_as_videos(masks, out_dir="Out/Ori/Masks", fps=4)
        break
        # 这里只跑一个 batch 做演示
 

