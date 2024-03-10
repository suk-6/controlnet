import torch
import warnings
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import OpenposeDetector

warnings.filterwarnings(action="ignore")

original = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
pose = openpose(original)

controlnet = [
    ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

prompt = "A postmark standing on a cloud in the sky, top quality"

result = pipe(
    prompt,
    [pose],
    num_inference_steps=20,
    generator=torch.Generator(device="cpu").manual_seed(1),
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    controlnet_conditioning_scale=1.0,
).images[0]

original.save("./original.png")
pose.save("./pose.png")
result.save("./output.png")
