import time
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image
from check_backend import *

check_backend()

device = "mps"
print(f"正在使用 {device} 進行加速... 🚀")

# 1. 这是我们选择的 DiT-XL 模型，并将使用半精度来节省 VRAM
model_id = "facebook/DiT-XL-2-256"

# 2. 创建 DiT pipeline
#    在 Apple Silicon 上，我们通常直接用 float32 精度，
#    但为了在 8GB VRAM 内跑起 XL 模型，我们必须使用 float16
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16,  cache_dir='./model/')

# 3. 选择一个采样器
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 4. 把整个 pipeline 移到你的 MPS 设备上
pipe = pipe.to(device)

print("模型載入完成，準備開始生成圖片... (ง •̀_•́)ง")

# 5. 执行生成！
#    class_labels=[281] 是 "tabby cat"（虎斑猫）
torch.mps.synchronize()
start_time = time.perf_counter()
image = pipe(class_labels=[281], num_inference_steps=25).images[0]
torch.mps.synchronize()
generation_time = time.perf_counter() - start_time

print(f"图片生成耗时：{generation_time:.2f} 秒")

# 6. 保存图片
output_path = "./image/dit_generated_image.png"
image.save(output_path)

print(f"圖片生成完畢！🎉 已經存到 {output_path}")
print("快去看看你的 M2 Pro 產的虎斑貓吧！")
