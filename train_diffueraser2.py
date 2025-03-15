#!/usr/bin/env python
# coding=utf-8

import argparse

import logging
import math
import os

import yaml
import shutil
from pathlib import Path

from einops import rearrange, repeat
import accelerate
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from huggingface_hub import create_repo
from packaging import version

from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import imageio
import diffusers
from diffusers import (
    AutoencoderKL,
    UniPCMultistepScheduler,
    DDPMScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from libs.unet_motion_model import UNetMotionModel
from libs.brushnet_CA import BrushNetModel

from libs.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from dataset2 import VideoMaskDataset
from diffusers.utils.torch_utils import randn_tensor
import cv2
import copy
import gc
from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a BrushNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="weights/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--brushnet_model_name_or_path",
        type=str,
        default="weights/diffuEraser",
        help="Path to pretrained brushnet model or model identifier from huggingface.co/models."
        " If not specified brushnet weights are initialized from unet.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="brushnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=114514, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input videos, all the videos in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames in each training video clip.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="VideoSelection",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--mask_data_dir",
        type=str,
        default="MaskSelection",
        help=(
            "A folder containing the mask data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_Unet_brushnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')

        # 新增：添加--config参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file "
    )

    args, unknown = parser.parse_known_args(input_args)

    # 如果指定了配置文件，加载并覆盖默认值
    if args.config:
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
        
        config_args = []
        for key, value in config_data.items():
            if isinstance(value, bool):
                # 处理布尔类型（例如 use_8bit_adam: true -> --use_8bit_adam）
                if value:
                    config_args.append(f"--{key}")
            else:
                config_args.extend([f"--{key}", str(value)])
        
        # 合并配置文件和命令行参数（命令行优先级更高）
        args = parser.parse_args(config_args + unknown)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the brushnet encoder."
        )

    return args

from PIL import Image
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames

def read_mask(validation_mask, n_total_frames, img_size, frames, mask_dilation_iter=4):
    cap = cv2.VideoCapture(validation_mask)
    if not cap.isOpened():
        print("Error: Could not open mask video.")
        exit()
    masks = []
    masked_images = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:  
            break
        if(idx >= n_total_frames):
            break
        mask = Image.fromarray(frame[...,::-1]).convert('L')
        if mask.size != img_size:
            mask = mask.resize(img_size, Image.NEAREST)
        mask = np.asarray(mask)
        m = np.array(mask > 0).astype(np.uint8)
        m = cv2.erode(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1)
        m = cv2.dilate(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=mask_dilation_iter)

        mask = Image.fromarray(m * 255)
        masks.append(mask)

        masked_image = np.array(frames[idx])*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        masked_images.append(masked_image)

        idx += 1
    cap.release()

    return masks, masked_images

def read_priori(priori, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    prioris=[]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if(idx >= n_total_frames):
            break
        img = Image.fromarray(frame[...,::-1])
        if img.size != img_size:
            img = img.resize(img_size)
        prioris.append(img)
        idx += 1
    cap.release()

    # os.remove(priori) # remove priori 
    return prioris

def read_video(validation_image, video_length, nframes, max_img_size=1280):
    vframes, aframes, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec', end_pts=video_length) # RGB
    fps = info['video_fps']
    n_total_frames = int(video_length * fps)
    n_clip = int(np.ceil(n_total_frames/nframes))

    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    max_size = max(frames[0].size)
    if(max_size<256):
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if(max_size>4096):
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")
    if max_size>max_img_size:
        ratio = max_size/max_img_size
        ratio_size = (int(frames[0].size[0]/ratio),int(frames[0].size[1]/ratio))
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    elif (frames[0].size[0]%8==0) and (frames[0].size[1]%8==0):
        img_size = frames[0].size
        resize_flag=False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    if resize_flag:
        frames = resize_frames(frames, img_size)
        img_size = frames[0].size

    return frames,n_clip, n_total_frames,img_size


def validation(
    vae, text_encoder, tokenizer, unet, brushnet, args, accelerator, weight_dtype, step, nframes=22
):
    unet = accelerator.unwrap_model(unet)
    print(f"Running validation...step_{step}.mp4")

    pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            brushnet=brushnet
        )
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = ""
    validation_images = "Validation/video_0.mp4"
    validation_masks = "Validation/mask_0.mp4"
    validation_ppp= "Validation/priori_2.mp4"

    ################ read input video ################ 
    
    frames, n_clip, n_total_frames,img_size = read_video(validation_images, video_length=10 , nframes=nframes, max_img_size=1280)
    video_len = len(frames)

    ################     read mask    ################
    validation_masks_input, validation_images_input = read_mask(validation_masks, video_len, img_size,  frames)

    ################    read priori   ################
    prioris = read_priori(validation_ppp, n_total_frames, img_size)

    ## recheck
    n_total_frames = min(min(len(frames), len(validation_masks_input)), len(prioris))
    validation_masks_input = validation_masks_input[:n_total_frames]
    validation_images_input = validation_images_input[:n_total_frames]
    frames = frames[:n_total_frames]
    prioris = prioris[:n_total_frames]

    prioris = resize_frames(prioris)
    validation_masks_input = resize_frames(validation_masks_input)
    validation_images_input = resize_frames(validation_images_input)
    resized_frames = resize_frames(frames)
    generator = torch.Generator(device=unet.device)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)  

    ## random noise
    real_video_length = len(validation_images_input)
    tar_width, tar_height = validation_images_input[0].size 

    shape = (
        nframes,
        4,
        tar_height//8,
        tar_width//8
    )
    if text_encoder is not None:
        prompt_embeds_dtype =text_encoder.dtype

    noise_pre = randn_tensor(shape, device=torch.device(unet.device), dtype=prompt_embeds_dtype, generator=generator) 
    noise = repeat(noise_pre, "t c h w->(repeat t) c h w", repeat=n_clip)[:real_video_length,...]

    ################  prepare priori  ################
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    images_preprocessed = []

    for image in prioris:
        # print(image.size) #(1152, 480)
        image = image_processor.preprocess(image, height=tar_height, width=tar_width).to(dtype=torch.float32)
        image = image.to(device=torch.device(unet.device), dtype=torch.float16)
        # print(image.shape) #torch.Size([1, 3, 480, 1152])
        images_preprocessed.append(image)
    pixel_values = torch.cat(images_preprocessed)

    with torch.no_grad():
        pixel_values = pixel_values.to(dtype=torch.float16)
        latents = []
        num=4
        for i in range(0, pixel_values.shape[0], num):
            latents.append(vae.encode(pixel_values[i : i + num]).latent_dist.sample())
        latents = torch.cat(latents, dim=0)
    latents = latents * vae.config.scaling_factor #[(b f), c1, h, w], c1=4
    
    torch.cuda.empty_cache()  
    timesteps = torch.tensor([0], device=unet.device)
    timesteps = timesteps.long()
    validation_masks_input_ori = copy.deepcopy(validation_masks_input)
    resized_frames_ori = copy.deepcopy(resized_frames)
    ################  Frame-by-frame inference  ################
    
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps) 
    latents = noisy_latents
    with torch.no_grad():
        images = pipeline(
            num_frames=nframes, 
            prompt=validation_prompts, 
            images=validation_images_input, 
            masks=validation_masks_input, 
            num_inference_steps=20, 
            generator=generator,
            guidance_scale=0,
            latents=latents,
        ).frames
    images = images[:real_video_length]

    gc.collect()
    torch.cuda.empty_cache()

    ################ Compose ################
    binary_masks = validation_masks_input_ori
    mask_blurreds = []

    # blur, you can adjust the parameters for better performance
    for i in range(len(binary_masks)):
        mask_blurred = cv2.GaussianBlur(np.array(binary_masks[i]), (21, 21), 0)/255.
        binary_mask = 1-(1-np.array(binary_masks[i])/255.) * (1-mask_blurred)
        mask_blurreds.append(Image.fromarray((binary_mask*255).astype(np.uint8)))
    binary_masks = mask_blurreds
    
    comp_frames = []
    for i in range(len(images)):
        mask = np.expand_dims(np.array(binary_masks[i]),2).repeat(3, axis=2).astype(np.float32)/255.
        img = (np.array(images[i]).astype(np.uint8) * mask \
            + np.array(resized_frames_ori[i]).astype(np.uint8) * (1 - mask)).astype(np.uint8)
        comp_frames.append(Image.fromarray(img))
    print("Compose Start")
    default_fps = 10
    writer = cv2.VideoWriter(f"Validation/step_{step}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                        default_fps, comp_frames[0].size)
    
    for f in range(real_video_length):
        img = np.array(comp_frames[f]).astype(np.uint8)
        writer.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    writer.release()
    print("Validation End")


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="scheduler",
                prediction_type="epsilon",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
                )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_path
    )
    unet = UNetMotionModel.from_pretrained(
        args.brushnet_model_name_or_path, subfolder="unet_main")

    if args.brushnet_model_name_or_path:
        logger.info("Loading existing brushnet weights")
        brushnet = BrushNetModel.from_pretrained(args.brushnet_model_name_or_path,subfolder="brushnet")
    else:
        logger.info("Initializing brushnet weights from unet")
        brushnet = BrushNetModel.from_unet(unet)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "unet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetMotionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    brushnet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = VideoMaskDataset(
        root_dir=args.train_data_dir, 
        mask_dir=args.mask_data_dir,
        sample_size=512, 
        sample_stride=1, 
        sample_n_frames=args.num_frames,
        mask_dilation_iter=3
    )
    train_dataset_len= len(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
    )
    train_dataloader_len=train_dataset_len//args.train_batch_size


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, brushnet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    brushnet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num batches each epoch = {train_dataloader_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path),map_location="cpu")
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    #一次性计算空输入编码
    empty_input = [""] * args.train_batch_size

    empty_prompt_ids = tokenizer(
        empty_input, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(accelerator.device)

    empty_encoder_hidden_states = text_encoder(empty_prompt_ids, return_dict=False)[0]
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):

            if  global_step in [10,50,100,200,300,400,600,700,800,900] or (accelerator.is_main_process and (global_step+1) % 500 == 0 and global_step!=0):
                validation(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
                           brushnet=brushnet, args=args, accelerator=accelerator, weight_dtype=weight_dtype, step=global_step, nframes=args.num_frames)

            with accelerator.accumulate(unet):
                with torch.no_grad():
                    #prepare latents
                    ppp_values = batch["pixel_values"].to(dtype=weight_dtype)
                    conditioning_pixel_values=batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                    masks=batch["masks"].to(dtype=weight_dtype)

                    #输入vae前的预处理
                    for b in range(ppp_values.shape[0]):  # 遍历 batch 维度
                        for t in range(ppp_values.shape[1]):  # 遍历时间维度（帧数）
                            # 取出当前批次 `b` 的第 `t` 帧图像,(0,255)→(-1,1)
                            frame = ppp_values[b, t]/255  # 形状 (3, H, W)
                            mask = masks[b, t]/255  # 形状 (1, H, W)
                            masked_frame = conditioning_pixel_values[b, t]/255
                            ppp_values[b, t]=(frame-0.5)/0.5
                            masks[b, t]=(mask-0.5)/0.5
                            conditioning_pixel_values[b, t]=(masked_frame-0.5)/0.5

                    video_length = ppp_values.shape[1]
                    ppp_values = rearrange(ppp_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(ppp_values).latent_dist.sample() * vae.config.scaling_factor

                    #prepare condition latent
                    video_length = conditioning_pixel_values.shape[1]
                    conditioning_pixel_values = rearrange(conditioning_pixel_values, "b f c h w -> (b f) c h w")
                    conditioning_latents = vae.encode(conditioning_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    conditioning_latents = rearrange(conditioning_latents, "(b f) c h w -> b f c h w", f=video_length)

                #prepare mask
                masks = rearrange(masks, "b f c h w -> (b f) c h w")
                masks = torch.nn.functional.interpolate(
                    masks, 
                    size=(
                        latents.shape[-2], 
                        latents.shape[-1]
                    )
                ).to(dtype=weight_dtype)
                masks = rearrange(masks, "(b f) c h w -> b f c h w", f=video_length)
                conditioning_latents=torch.concat([conditioning_latents,masks],2)
                conditioning_latents=rearrange(conditioning_latents, "b f c h w -> (b f) c h w")
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = args.train_batch_size
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                #prompt空输入
                brushnet_prompt_embeds = rearrange(repeat(empty_encoder_hidden_states, "b c d -> b t c d", t=video_length), 'b t c d -> (b t) c d')
                # print(f"noisy_latents:{noisy_latents.shape}")
                # print(f"timesteps:{timesteps}")
                # print(f"brushnet_prompt_embeds:{brushnet_prompt_embeds.shape}")
                # print(f"conditioning_latents:{conditioning_latents.shape}")
                # noisy_latents:torch.Size([22, 4, 64, 64])
                # timesteps:tensor([667], device='cuda:0')
                # brushnet_prompt_embeds:torch.Size([22, 77, 768])
                # conditioning_latents:torch.Size([22, 5, 64, 64])

                
                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample, up_block_res_samples = brushnet(
                        noisy_latents.to(dtype=weight_dtype),
                        timesteps,
                        encoder_hidden_states=brushnet_prompt_embeds,
                        brushnet_cond=conditioning_latents,
                        return_dict=False,
                    )

                # print(f"latents:{latents.shape}")
                # print(f"timesteps:{timesteps}")
                # print(f"prompt_embeds:{empty_encoder_hidden_states.shape}")
                # print(f"down_block_res_samples:{len(down_block_res_samples)},{down_block_res_samples[0].shape}")
                # print(f"mid_block_res_sample:{mid_block_res_sample.shape}")
                # print(f"up_block_res_samples:{len(up_block_res_samples)},{up_block_res_samples[0].shape}")
                # latents:torch.Size([22, 4, 64, 64])
                # timesteps:tensor([149], device='cuda:0')
                # prompt_embeds:torch.Size([1, 77, 768])
                # down_block_res_samples:12,torch.Size([22, 320, 64, 64])
                # mid_block_res_sample:torch.Size([22, 1280, 8, 8])
                # up_block_res_samples:15,torch.Size([22, 1280, 8, 8])

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_encoder_hidden_states,
                    down_block_add_samples=down_block_res_samples,
                    mid_block_add_sample=mid_block_res_sample,
                    up_block_add_samples=up_block_res_samples,
                    return_dict=False,
                    num_frames=video_length,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 正确保存UNet
        unet = unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
