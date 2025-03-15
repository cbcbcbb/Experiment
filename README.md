# Experiment
## 模型与数据集
压缩包已包含所有预训练模型与高质量验证数据集，解压即可。换了个数据集结构用了新的加载，使用 `train_diffueraser2.py` 

## 环境配置
```bash
cd DiffuEraser
conda create -n experiment pytnon=3.10
conda activate experiment
pip install -r requirements.txt
```
## 运行
```bash
accelerate launch --mixed_precision fp16  --main_process_port 29501  train_diffueraser2.py --config Configs/default.yaml 
```
配置文件使用默认参数
```
gradient_checkpointing: true  
num_frames: 22                
gradient_accumulation_steps: 2
use_8bit_adam: true           
mixed_precision: "fp16"
```
可直接通过命令行覆盖
```bash
accelerate launch --mixed_precision fp16  --main_process_port 29501  train_diffueraser2.py --config Configs/default.yaml --num_frames 22  
```

## 监看训练
```bash
tensorboard --logdir brushnet-model/logs
```

## 定期可视化结果
Validation文件夹中有固定步数执行的可视化评估，前1000步密集，后面每500步推理一次，模型训练也是500步保存一次checkpoint
