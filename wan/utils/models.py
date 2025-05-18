import os
import torch

from pathlib import Path
from mmgp import offload

import wan
from wan.configs import WAN_CONFIGS

wan_choices_t2v = ["ckpts/wan2.1_text2video_1.3B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_bf16.safetensors",
                   "ckpts/wan2.1_text2video_14B_quanto_int8.safetensors", "ckpts/wan2.1_Vace_1.3B_mbf16.safetensors",
                   "ckpts/wan2.1_recammaster_1.3B_bf16.safetensors",
                   "ckpts/sky_reels2_diffusion_forcing_1.3B_mbf16.safetensors",
                   "ckpts/sky_reels2_diffusion_forcing_14B_bf16.safetensors",
                   "ckpts/sky_reels2_diffusion_forcing_14B_quanto_int8.safetensors",
                   "ckpts/sky_reels2_diffusion_forcing_720p_14B_mbf16.safetensors",
                   "ckpts/sky_reels2_diffusion_forcing_720p_14B_quanto_mbf16_int8.safetensors",
                   "ckpts/wan2_1_phantom_1.3B_mbf16.safetensors", "ckpts/wan2.1_Vace_14B_mbf16.safetensors",
                   "ckpts/wan2.1_Vace_14B_quanto_mbf16_int8.safetensors"]
wan_choices_i2v = ["ckpts/wan2.1_image2video_480p_14B_mbf16.safetensors",
                   "ckpts/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors",
                   "ckpts/wan2.1_image2video_720p_14B_mbf16.safetensors",
                   "ckpts/wan2.1_image2video_720p_14B_quanto_mbf16_int8.safetensors",
                   "ckpts/wan2.1_Fun_InP_1.3B_bf16.safetensors", "ckpts/wan2.1_Fun_InP_14B_bf16.safetensors",
                   "ckpts/wan2.1_Fun_InP_14B_quanto_int8.safetensors", "ckpts/wan2.1_FLF2V_720p_14B_bf16.safetensors",
                   "ckpts/wan2.1_FLF2V_720p_14B_quanto_int8.safetensors",
                   "ckpts/wan2.1_fantasy_speaking_14B_bf16.safetensors"]
ltxv_choices = ["ckpts/ltxv_0.9.7_13B_dev_bf16.safetensors", "ckpts/ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors",
                "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors"]

hunyuan_choices = ["ckpts/hunyuan_video_720_bf16.safetensors", "ckpts/hunyuan_video_720_quanto_int8.safetensors",
                   "ckpts/hunyuan_video_i2v_720_bf16.safetensors",
                   "ckpts/hunyuan_video_i2v_720_quanto_int8v2.safetensors",
                   "ckpts/hunyuan_video_custom_720_bf16.safetensors",
                   "ckpts/hunyuan_video_custom_720_quanto_bf16_int8.safetensors"]

model_signatures = {"t2v": "text2video_14B", "t2v_1.3B": "text2video_1.3B", "fun_inp_1.3B": "Fun_InP_1.3B",
                    "fun_inp": "Fun_InP_14B",
                    "i2v": "image2video_480p", "i2v_720p": "image2video_720p", "vace_1.3B": "Vace_1.3B",
                    "vace_14B": "Vace_14B", "recam_1.3B": "recammaster_1.3B",
                    "flf2v_720p": "FLF2V_720p", "sky_df_1.3B": "sky_reels2_diffusion_forcing_1.3B",
                    "sky_df_14B": "sky_reels2_diffusion_forcing_14B",
                    "sky_df_720p_14B": "sky_reels2_diffusion_forcing_720p_14B",
                    "phantom_1.3B": "phantom_1.3B", "fantasy": "fantasy", "ltxv_13B": "ltxv_0.9.7_13B_dev",
                    "ltxv_13B_distilled": "ltxv_0.9.7_13B_distilled", "hunyuan": "hunyuan_video_720",
                    "hunyuan_i2v": "hunyuan_video_i2v_720", "hunyuan_custom": "hunyuan_video_custom"}

transformer_choices = wan_choices_t2v + wan_choices_i2v + ltxv_choices + hunyuan_choices

server_config = {"attention_mode": "auto",
                 "transformer_types": [],
                 "transformer_quantization": "int8",
                 "text_encoder_quantization": "int8",
                 "save_path": "outputs",
                 "compile": "",
                 "metadata_type": "metadata",
                 "default_ui": "t2v",
                 "boost": 1,
                 "clear_file_list": 5,
                 "vae_config": 0,
                 "preload_model_policy": [],
                 "UI_theme": "default"}

text_encoder_quantization = server_config.get("text_encoder_quantization", "int8")
transformer_quantization = server_config.get("transformer_quantization", "int8")
transformer_dtype_policy = server_config.get("transformer_dtype_policy", "")

quantizeTransformer = True if transformer_quantization == "int8" else False

major, minor = torch.cuda.get_device_capability()
if major < 8:
    print("Switching to FP16 models when possible as GPU architecture doesn't support optimed BF16 Kernels")
    bfloat16_supported = False
else:
    bfloat16_supported = True


def test_class_i2v(model_filename):
    return "image2video" in model_filename or "Fun_InP" in model_filename or "FLF2V" in model_filename or "fantasy" in model_filename or "hunyuan_video_i2v" in model_filename


def get_transformer_dtype(model_family, transformer_dtype_policy):
    if len(transformer_dtype_policy) == 0:
        if not bfloat16_supported:
            return torch.float16
        else:
            if model_family == "wan" and False:
                return torch.float16
            else:
                return torch.bfloat16
        return transformer_dtype
    elif transformer_dtype_policy == "fp16":
        return torch.float16
    else:
        return torch.bfloat16


def get_model_family(model_filename):
    if "wan" in model_filename or "sky" in model_filename:
        return "wan"
    elif "ltxv" in model_filename:
        return "ltxv"
    elif "hunyuan" in model_filename:
        return "hunyuan"
    else:
        raise Exception(f"Unknown model family for model'{model_filename}'")


def get_model_type(model_filename):
    for model_type, signature in model_signatures.items():
        if signature in model_filename:
            return model_type
    raise Exception("Unknown model:" + model_filename)


def get_wan_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
    if text_encoder_quantization == "int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_int8")
    return text_encoder_filename


def get_ltxv_text_encoder_filename(text_encoder_quantization):
    text_encoder_filename = "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors"
    if text_encoder_quantization == "int8":
        text_encoder_filename = text_encoder_filename.replace("bf16", "quanto_bf16_int8")
    return text_encoder_filename


def get_hunyuan_text_encoder_filename(text_encoder_quantization):
    if text_encoder_quantization == "int8":
        text_encoder_filename = "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"
    else:
        text_encoder_filename = "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"

    return text_encoder_filename


def get_model_manager(model_family):
    if model_family == "wan":
        return None
    elif model_family == "ltxv":
        from ltxv import model_def
        return model_def
    else:
        raise Exception("model family not supported")


def get_model_filename(model_type, quantization="int8", dtype_policy=""):
    signature = model_signatures[model_type]
    choices = [name for name in transformer_choices if signature in name]
    if len(quantization) == 0:
        quantization = "bf16"

    model_family = get_model_family(choices[0])
    dtype = get_transformer_dtype(model_family, dtype_policy)
    if len(choices) <= 1:
        raw_filename = choices[0]
    else:
        sub_choices = [name for name in choices if quantization in name]
        if len(sub_choices) > 0:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"
            new_sub_choices = [name for name in sub_choices if dtype_str in name]
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 else sub_choices
            raw_filename = sub_choices[0]
        else:
            raw_filename = choices[0]

    if dtype == torch.float16 and not "fp16" in raw_filename and model_family == "wan":
        if "quanto_int8" in raw_filename:
            raw_filename = raw_filename.replace("quanto_int8", "quanto_fp16_int8")
        elif "quanto_bf16_int8" in raw_filename:
            raw_filename = raw_filename.replace("quanto_bf16_int8", "quanto_fp16_int8")
        elif "quanto_mbf16_int8" in raw_filename:
            raw_filename = raw_filename.replace("quanto_mbf16_int8", "quanto_mfp16_int8")
    return raw_filename


def download_models(transformer_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos + 1:]
        return [filename]

    def process_files_def(repoId, sourceFolderList, fileList):
        targetRoot = "ckpts/"
        for sourceFolder, files in zip(sourceFolderList, fileList):
            if len(files) == 0:
                if not Path(targetRoot + sourceFolder).exists():
                    snapshot_download(repo_id=repoId, allow_patterns=sourceFolder + "/*", local_dir=targetRoot)
            else:
                for onefile in files:
                    if len(sourceFolder) > 0:
                        if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile):
                            hf_hub_download(repo_id=repoId, filename=onefile, local_dir=targetRoot,
                                            subfolder=sourceFolder)
                    else:
                        if not os.path.isfile(targetRoot + onefile):
                            hf_hub_download(repo_id=repoId, filename=onefile, local_dir=targetRoot)

    from huggingface_hub import hf_hub_download, snapshot_download

    shared_def = {
        "repoId": "DeepBeepMeep/Wan2.1",
        "sourceFolderList": ["pose", "depth", "mask", "wav2vec", ""],
        "fileList": [[], [], ["sam_vit_h_4b8939_fp16.safetensors"],
                     ["config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json",
                      "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
                     ["flownet.pkl"]]
    }
    process_files_def(**shared_def)

    model_family = get_model_family(transformer_filename)
    if model_family == "wan":
        text_encoder_filename = get_wan_text_encoder_filename(text_encoder_quantization)
        model_def = {
            "repoId": "DeepBeepMeep/Wan2.1",
            "sourceFolderList": ["xlm-roberta-large", "umt5-xxl", ""],
            "fileList": [
                ["models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", "sentencepiece.bpe.model",
                 "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"],
                ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"] + computeList(
                    text_encoder_filename),
                ["Wan2.1_VAE.safetensors", "fantasy_proj_model.safetensors"] + computeList(transformer_filename)]
        }
    elif model_family == "ltxv":
        text_encoder_filename = get_ltxv_text_encoder_filename(text_encoder_quantization)
        model_def = {
            "repoId": "DeepBeepMeep/LTX_Video",
            "sourceFolderList": ["T5_xxl_1.1", ""],
            "fileList": [
                ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"] + computeList(
                    text_encoder_filename), ["ltxv_0.9.7_VAE.safetensors", "ltxv_0.9.7_spatial_upscaler.safetensors",
                                             "ltxv_scheduler.json"] + computeList(transformer_filename)]
        }
    elif model_family == "hunyuan":
        text_encoder_filename = get_hunyuan_text_encoder_filename(text_encoder_quantization)
        model_def = {
            "repoId": "DeepBeepMeep/HunyuanVideo",
            "sourceFolderList": ["llava-llama-3-8b", "clip_vit_large_patch14", ""],
            "fileList": [["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
                          "preprocessor_config.json"] + computeList(text_encoder_filename),
                         ["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json",
                          "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                         ["hunyuan_video_720_quanto_int8_map.json", "hunyuan_video_custom_VAE_fp32.safetensors",
                          "hunyuan_video_custom_VAE_config.json", "hunyuan_video_VAE_fp32.safetensors",
                          "hunyuan_video_VAE_config.json", "hunyuan_video_720_quanto_int8_map.json"] + computeList(
                             transformer_filename)]
        }

    else:
        model_manager = get_model_manager(model_family)
        model_def = model_manager.get_files_def(transformer_filename, text_encoder_quantization)

    process_files_def(**model_def)


def get_dependent_models(model_filename, quantization, dtype_policy):
    if "fantasy" in model_filename:
        return [get_model_filename("i2v_720p", quantization, dtype_policy)]
    elif "ltxv_0.9.7_13B_distilled_lora128" in model_filename:
        return [get_model_filename("ltxv_13B", quantization, dtype_policy)]
    else:
        return []


def load_wan_model(model_filename, quantizeTransformer=False, dtype=torch.bfloat16, VAE_dtype=torch.float32,
                   mixed_precision_transformer=False):
    filename = model_filename[-1]
    print(f"Loading '{filename}' model...")

    if test_class_i2v(model_filename[0]):
        cfg = WAN_CONFIGS['i2v-14B']
        model_factory = wan.WanI2V
    else:
        cfg = WAN_CONFIGS['t2v-14B']
        # cfg = WAN_CONFIGS['t2v-1.3B']
        if get_model_type(filename) in ("sky_df_1.3B", "sky_df_14B", "sky_df_720p_14B"):
            model_factory = wan.DTT2V
        else:
            model_factory = wan.WanT2V

    wan_model = model_factory(
        config=cfg,
        checkpoint_dir="ckpts",
        model_filename=model_filename,
        text_encoder_filename=get_wan_text_encoder_filename(text_encoder_quantization),
        quantizeTransformer=quantizeTransformer,
        dtype=dtype,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=mixed_precision_transformer
    )

    pipe = {"transformer": wan_model.model, "text_encoder": wan_model.text_encoder.model, "vae": wan_model.vae.model}
    if hasattr(wan_model, "clip"):
        pipe["text_encoder_2"] = wan_model.clip.model
    return wan_model, pipe


def load_ltxv_model(model_filename, quantizeTransformer=False, dtype=torch.bfloat16, VAE_dtype=torch.float32,
                    mixed_precision_transformer=False):
    filename = model_filename[-1]
    print(f"Loading '{filename}' model...")
    from ltx_video.ltxv import LTXV

    ltxv_model = LTXV(
        model_filepath=model_filename,
        text_encoder_filepath=get_ltxv_text_encoder_filename(text_encoder_quantization),
        dtype=dtype,
        # quantizeTransformer = quantizeTransformer,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=mixed_precision_transformer
    )

    pipeline = ltxv_model.pipeline
    pipe = {"transformer": pipeline.video_pipeline.transformer, "vae": pipeline.vae,
            "text_encoder": pipeline.video_pipeline.text_encoder, "latent_upsampler": pipeline.latent_upsampler}

    return ltxv_model, pipe


def load_hunyuan_model(model_filename, quantizeTransformer=False, dtype=torch.bfloat16, VAE_dtype=torch.float32,
                       mixed_precision_transformer=False):
    filename = model_filename[-1]
    print(f"Loading '{filename}' model...")
    from hyvideo.hunyuan import HunyuanVideoSampler

    hunyuan_model = HunyuanVideoSampler.from_pretrained(
        model_filepath=model_filename,
        text_encoder_filepath=get_hunyuan_text_encoder_filename(text_encoder_quantization),
        dtype=dtype,
        # quantizeTransformer = quantizeTransformer,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=mixed_precision_transformer
    )

    pipe = {"transformer": hunyuan_model.model, "text_encoder": hunyuan_model.text_encoder,
            "text_encoder_2": hunyuan_model.text_encoder_2, "vae": hunyuan_model.vae}

    from hyvideo.modules.models import get_linear_split_map

    split_linear_modules_map = get_linear_split_map()
    hunyuan_model.model.split_linear_modules_map = split_linear_modules_map
    offload.split_linear_modules(hunyuan_model.model, split_linear_modules_map)

    return hunyuan_model, pipe


def load_models(model_filename, profile):
    global transformer_filename, transformer_loras_filenames
    model_family = get_model_family(model_filename)

    dependent_models = get_dependent_models(model_filename, quantization=transformer_quantization,
                                            dtype_policy=transformer_dtype_policy)
    new_transformer_loras_filenames = [model_filename] if "_lora" in model_filename else None
    model_filelist = dependent_models + [model_filename]
    for filename in model_filelist:
        download_models(filename)
    transformer_dtype = get_transformer_dtype(model_family, transformer_dtype_policy)
    VAE_dtype = torch.float16 if server_config.get("vae_precision", "16") == "16" else torch.float
    mixed_precision_transformer = server_config.get("mixed_precision", "0") == "1"
    transformer_filename = None
    transformer_loras_filenames = None
    new_transformer_filename = model_filelist[-1]
    if model_family == "wan":
        wan_model, pipe = load_wan_model(model_filelist, quantizeTransformer=quantizeTransformer,
                                         dtype=transformer_dtype, VAE_dtype=VAE_dtype,
                                         mixed_precision_transformer=mixed_precision_transformer)
    elif model_family == "ltxv":
        wan_model, pipe = load_ltxv_model(model_filelist, quantizeTransformer=quantizeTransformer,
                                          dtype=transformer_dtype, VAE_dtype=VAE_dtype,
                                          mixed_precision_transformer=mixed_precision_transformer)
    elif model_family == "hunyuan":
        wan_model, pipe = load_hunyuan_model(model_filelist, quantizeTransformer=quantizeTransformer,
                                             dtype=transformer_dtype, VAE_dtype=VAE_dtype,
                                             mixed_precision_transformer=mixed_precision_transformer)
    else:
        raise Exception(f"Model '{new_transformer_filename}' not supported.")
    wan_model._model_file_name = new_transformer_filename
    kwargs = {"extraModelsToQuantize": None}
    if profile in (2, 4, 5):
        kwargs["budgets"] = {"transformer": 100,
                             "text_encoder": 100,
                             "*": max(1000 if profile == 5 else 3000, 0)}
    elif profile == 3:
        kwargs["budgets"] = {"*": "70%"}

    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer

    prompt_enhancer_image_caption_model = None
    prompt_enhancer_image_caption_processor = None
    prompt_enhancer_llm_model = None
    prompt_enhancer_llm_tokenizer = None

    offloadobj = offload.profile(pipe, profile_no=profile, compile=compile, quantizeTransformer=quantizeTransformer,
                                 loras="transformer", coTenantsMap={}, convertWeightsFloatTo=transformer_dtype,
                                 **kwargs)

    transformer_filename = new_transformer_filename
    transformer_loras_filenames = new_transformer_loras_filenames
    return wan_model, offloadobj, pipe["transformer"]
