import argparse
import numpy as np
import os
import torch
from PIL import Image

from wan.utils.models import load_models
from wan.utils.utils import cache_video

TRANSFORMER_CHOICES = [
    # WAN models
    "ckpts/wan2.1_text2video_1.3B_bf16.safetensors",
    "ckpts/wan2.1_text2video_14B_bf16.safetensors",
    "ckpts/wan2.1_text2video_14B_quanto_int8.safetensors",
    "ckpts/wan2.1_image2video_480p_14B_mbf16.safetensors",
    "ckpts/wan2.1_image2video_720p_14B_mbf16.safetensors",
    "ckpts/wan2.1_Fun_InP_14B_bf16.safetensors",
    "ckpts/wan2.1_recammaster_1.3B_bf16.safetensors",
    "ckpts/wan2.1_Vace_14B_mbf16.safetensors",
    "ckpts/sky_reels2_diffusion_forcing_14B_bf16.safetensors",
    "ckpts/wan2_1_phantom_1.3B_mbf16.safetensors",
    "ckpts/wan2.1_FLF2V_720p_14B_bf16.safetensors",
    # LTXV
    "ckpts/ltxv_0.9.7_13B_dev_bf16.safetensors",
    "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors",
    # Hunyuan
    "ckpts/hunyuan_video_720_bf16.safetensors",
    "ckpts/hunyuan_video_i2v_720_bf16.safetensors",
    "ckpts/hunyuan_video_custom_720_bf16.safetensors",
]

MODEL_TYPES = {
    "wan": ["i2v", "t2v", "recam", "vace", "df", "phantom", "fantasy"],
    "ltxv": ["ltxv"],
    "hunyuan": ["hunyuan", "hunyuan_i2v", "hunyuan_custom"]
}


def parse_resolution(res_str):
    width, height = map(int, res_str.lower().split("x"))
    return width, height


def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Generate video using WAN2GP CLI")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=TRANSFORMER_CHOICES)
    parser.add_argument("--model-type", type=str, required=True, choices=sum(MODEL_TYPES.values(), []))
    parser.add_argument("--resolution", type=str, default="832x480")
    parser.add_argument("--video-length", type=int, default=81)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--image-start", type=str, help="Path to start image")
    parser.add_argument("--image-end", type=str, help="Path to end image")
    parser.add_argument("--video-guide", type=str, help="Path to video guide")
    parser.add_argument("--profile", action="store_true", help="mmgp profile", default=3)
    args = parser.parse_args()

    if args.image_start and args.video_guide:
        raise ValueError("Cannot use both --image-start and --video-guide. Please choose one.")

    width, height = parse_resolution(args.resolution)

    if args.seed < 0:
        args.seed = np.random.randint(0, 1e9)

    print("[INFO] Loading model with MMGP offload...")
    model, pipe, _ = load_models(args.model, args.profile)

    print("[INFO] Generating video...")
    torch.manual_seed(args.seed)
    image_start = load_image(args.image_start) if args.image_start else None
    image_end = load_image(args.image_end) if args.image_end else None
    video_guide = args.video_guide if args.video_guide else None

    video_tensor = model.generate(
        input_prompt=args.prompt,
        input_frames=None,
        input_ref_images=None,
        input_masks=None,
        source_video=video_guide,
        image_start=image_start,
        image_end=image_end,
        frame_num=(args.video_length // 4) * 4 + 1,
        height=height,
        width=width,
        sampling_steps=args.steps,
        guide_scale=args.guidance_scale,
        fit_into_canvas=False,
        shift=5.0,
        seed=args.seed,
        n_prompt="",
        callback=None,
        enable_RIFLEx=True,
        VAE_tile_size=128,
        joint_pass=True,
    )

    print("[INFO] Saving video...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cache_video(video_tensor[None], args.output, fps=16, nrow=1, normalize=True, value_range=(-1, 1))

    print(f"[DONE] Video saved to {args.output}")


if __name__ == "__main__":
    main()
