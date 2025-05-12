import argparse
import os
import torch
import numpy as np
from mmgp import offload
from wan import WanI2V, WanT2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from PIL import Image

SUPPORTED_MODELS = {
    "i2v": [
        "ckpts/wan2.1_image2video_480p_14B_mbf16.safetensors",
        "ckpts/wan2.1_image2video_720p_14B_mbf16.safetensors",
        "ckpts/wan2.1_Fun_InP_14B_bf16.safetensors",
        "ckpts/wan2.1_FLF2V_720p_14B_bf16.safetensors",
        "ckpts/wan2.1_FLF2V_720p_14B_quanto_int8.safetensors",
        "ckpts/wan2.1_fantasy_speaking_14B_bf16.safetensors"
    ],
    "t2v": [
        "ckpts/wan2.1_text2video_14B_bf16.safetensors",
        "ckpts/wan2.1_text2video_1.3B_bf16.safetensors"
    ],
    "vace": [
        "ckpts/wan2.1_Vace_1.3B_preview_mbf16.safetensors"
    ],
    "recam": [
        "ckpts/wan2.1_recammaster_1.3B_bf16.safetensors"
    ],
    "df": [
        "ckpts/sky_reels2_diffusion_forcing_1.3B_mbf16.safetensors",
        "ckpts/sky_reels2_diffusion_forcing_14B_bf16.safetensors",
        "ckpts/sky_reels2_diffusion_forcing_14B_quanto_int8.safetensors",
        "ckpts/sky_reels2_diffusion_forcing_720p_14B_mbf16.safetensors",
        "ckpts/sky_reels2_diffusion_forcing_720p_14B_quanto_mbf16_int8.safetensors"
    ],
    "phantom": [
        "ckpts/wan2_1_phantom_1.3B_mbf16.safetensors"
    ]
}

ALL_MODEL_TYPES = list(SUPPORTED_MODELS.keys())

def load_model(model_path, model_type="i2v", quantize=False, dtype=torch.float16, profile_no=3):
    config_key = 'i2v-14B' if "i2v" in model_type else 't2v-14B'
    config = WAN_CONFIGS[config_key]

    model_class = WanI2V if "i2v" in model_type else WanT2V
    model = model_class(
        config=config,
        checkpoint_dir="ckpts",
        model_filename=model_path,
        text_encoder_filename="ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors",
        quantizeTransformer=quantize,
        dtype=dtype,
        VAE_dtype=torch.float16
    )

    model._model_file_name = model_path
    pipe = {"transformer": model.model, "text_encoder": model.text_encoder.model}

    offload.profile(pipe, profile_no=profile_no, compile="", quantizeTransformer=quantize,
                    perc_reserved_mem_max=0.3, convertWeightsFloatTo=dtype)

    return model

def parse_resolution(res_str):
    width, height = map(int, res_str.lower().split("x"))
    return width, height

def load_image(path):
    image = Image.open(path).convert("RGB")
    return image

def main():
    parser = argparse.ArgumentParser(description="Generate video using WAN2GP model")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=ALL_MODEL_TYPES, default="i2v")
    parser.add_argument("--resolution", type=str, default="832x480")
    parser.add_argument("--video-length", type=int, default=81)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--image-start", type=str, help="Path to start image")
    parser.add_argument("--image-end", type=str, help="Path to end image")
    parser.add_argument("--video-guide", type=str, help="Path to video guide")
    parser.add_argument('--profile', type=int, default=3, help='Offload profile number')

    args = parser.parse_args()
    width, height = parse_resolution(args.resolution)

    if args.seed < 0:
        args.seed = np.random.randint(0, 1e9)

    if args.image_start and args.video_guide:
        raise ValueError("Cannot use both --image-start and --video-guide. Please choose one.")

    if args.image_end and "i2v" not in args.model_type:
        raise ValueError("--image-end is only supported with image-to-video (i2v) models.")

    if args.model not in SUPPORTED_MODELS[args.model_type]:
        raise ValueError(f"Invalid model file for type '{args.model_type}'.\nSupported: {SUPPORTED_MODELS[args.model_type]}")

    print("[INFO] Loading model...")
    model = load_model(args.model, model_type=args.model_type, profile_no=args.profile)

    print("[INFO] Generating video...")
    torch.manual_seed(args.seed)

    image_start = load_image(args.image_start) if args.image_start else None
    image_end = load_image(args.image_end) if args.image_end else None
    video_guide = args.video_guide if args.video_guide else None

    video_tensor = model.generate(
        prompt=args.prompt,
        input_frames=None,
        input_ref_images=None,
        input_masks=None,
        source_video=video_guide,
        image=image_start,
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
