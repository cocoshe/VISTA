import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from diffusers import BitsAndBytesConfig
from diffusers.utils import export_to_video

from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
import einops
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageFilter

from tqdm import tqdm
import spacy
import cv2
nlp = spacy.load("en_core_web_sm")

with open("/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/VSPW_480p/val_video_semantic.json", "r") as f:
    video_semantic_dict = json.load(f)

def apply_morphological_operations(attn_map, kernel_size=3):
    attn_map = np.array(attn_map)
    attn_map_uint8 = (attn_map * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_map = cv2.morphologyEx(attn_map_uint8, cv2.MORPH_CLOSE, kernel)
    cleaned_attn_map = closed_map.astype(np.float32) / 255.0
    return cleaned_attn_map

def build_attn_maps_v2(atten_dict_list, tokens, frames, height, width, 
                    vae_spatial_compression_ratio, vae_temporal_compression_ratio,
                    patch_size_ratio, patch_size_t_ratio):
    selected_tokens_idxs = []
    for token_idx, token in enumerate(tokens):
        if (nlp(token)[0].pos_ == 'NOUN'):
            selected_tokens_idxs.append(token_idx)
    selected_tokens_idxs = np.array(selected_tokens_idxs)
    scaled_frames = ((frames + 1) // vae_temporal_compression_ratio + 1) // patch_size_t_ratio
    scaled_height = height // vae_spatial_compression_ratio // patch_size_ratio
    scaled_width = width // vae_spatial_compression_ratio // patch_size_ratio
    n_timesteps = len(atten_dict_list)
    n_blks = len(atten_dict_list[0])
    # all_attn_maps = []
    ts_blks_video_emb = []
    ts_blks_text_emb = []
    for timestep_idx in tqdm(range(n_timesteps)):
        # ts_attn_maps = []
        blocks_video_emb = []
        blocks_text_emb = []
        for blk_idx in range(n_blks):
            video_emb = atten_dict_list[timestep_idx][blk_idx]["video_emb"]
            text_emb = atten_dict_list[timestep_idx][blk_idx]["text_emb"]
            video_emb = einops.rearrange(video_emb, "b (t h w) head c -> b t h w head c", t=scaled_frames,
                                          h=scaled_height, w=scaled_width)
            # attn_maps = einops.einsum(  # seq_len, frames, h, w
            #     video_emb,
            #     text_emb,
            #     "b t h w head c, b s head c -> s t h w",
            # )
            blocks_video_emb.append(video_emb)
            blocks_text_emb.append(text_emb)
            # ts_attn_maps.append(attn_maps)
        blocks_video_emb = np.stack(blocks_video_emb)  # n_blks, b, t, h, w, head, c
        blocks_text_emb = np.stack(blocks_text_emb)  # n_blks, b, s, head, c
        # ts_attn_maps = np.stack(ts_attn_maps)  # n_blks, seq_len, t, h, w        
        # all_attn_maps.append(ts_attn_maps)
        ts_blks_video_emb.append(blocks_video_emb)
        ts_blks_text_emb.append(blocks_text_emb)
    ts_blks_video_emb = np.stack(ts_blks_video_emb)  # n_timesteps, n_blks, b, t, h, w, head, c
    ts_blks_text_emb = np.stack(ts_blks_text_emb)  # n_timesteps, n_blks, b, s, head, c
    all_attn_maps = einops.einsum(  # n_timesteps, n_blks, seq_len, t, h, w
        ts_blks_video_emb,
        ts_blks_text_emb,
        "nt nb b t h w head c, nt nb b s head c -> nt nb b s t h w",
    )
    # import pdb
    # pdb.set_trace()
    all_attn_maps = all_attn_maps[:, :, :, selected_tokens_idxs]  # n_timesteps, n_blks, bs, seq_len, t, h, w
    # all_attn_maps_softmax = torch.softmax(torch.from_numpy(all_attn_maps), 3).numpy()  # n_timesteps, n_blks, bs, seq_len, t, head, h, w
    all_attn_maps_softmax = all_attn_maps
    all_attn_maps_softmax = einops.reduce(
        all_attn_maps_softmax, "nt nb b s t h w -> s t h w", reduction="mean"
    )

    tokens = [tokens[i] for i in selected_tokens_idxs]
    n_tokens = len(tokens)
    # assert n_tokens == attn_maps.shape[0], f"Token length {len(tokens)} does not match attention map length {attn_maps.shape[0]}"
    assert n_tokens == all_attn_maps_softmax.shape[0], f"Token length {len(tokens)} does not match attention map length {all_attn_maps_softmax.shape[0]}"

    all_attn_maps = all_attn_maps_softmax
    all_attn_maps_min = all_attn_maps.min()
    all_attn_maps_max = all_attn_maps.max()
    # Convert to a matplotlib color scheme
    attn_map_out = []
    for frame_idx in range(all_attn_maps.shape[1]):
        frame_attn_map = all_attn_maps[:, frame_idx]  # n_tokens, t, h, w
        frame_attn_out = []
        for attn_map in frame_attn_map:
            attn_map = (attn_map - all_attn_maps_min) / (all_attn_maps_max - all_attn_maps_min)
            attn_map = plt.get_cmap('plasma')(attn_map)
            attn_map = (attn_map[:, :, :3] * 255).astype(np.uint8)
            frame_attn_out.append(attn_map)
        attn_map_out.append(frame_attn_out)  # attn_map_out: t, n_tokens, h, w

    output = {}
    for token_idx, token in enumerate(tokens):
        output[token] = []
        for frame_idx in range(len(attn_map_out)):
            attn_map = attn_map_out[frame_idx][token_idx]
            output[token].append(PIL.Image.fromarray(attn_map))
    
    return output


def build_attn_maps(atten_dict_list, tokens, frames, height, width, 
                    vae_spatial_compression_ratio, vae_temporal_compression_ratio,
                    patch_size_ratio, patch_size_t_ratio):
    selected_tokens_idxs = []
    for token_idx, token in enumerate(tokens):
        if (nlp(token)[0].pos_ == 'NOUN'):
            selected_tokens_idxs.append(token_idx)
    selected_tokens_idxs = np.array(selected_tokens_idxs)
    scaled_frames = ((frames + 1) // vae_temporal_compression_ratio + 1) // patch_size_t_ratio
    scaled_height = height // vae_spatial_compression_ratio // patch_size_ratio
    scaled_width = width // vae_spatial_compression_ratio // patch_size_ratio
    n_timesteps = len(atten_dict_list)
    n_blks = len(atten_dict_list[0])
    all_attn_maps = []
    for timestep_idx in tqdm(range(n_timesteps)):
        ts_attn_maps = []
        for blk_idx in range(n_blks):
            video_emb = atten_dict_list[timestep_idx][blk_idx]["video_emb"]
            text_emb = atten_dict_list[timestep_idx][blk_idx]["text_emb"]
            video_emb = einops.rearrange(video_emb, "b (t h w) head c -> b t h w head c", t=scaled_frames,
                                          h=scaled_height, w=scaled_width)
            attn_maps = einops.einsum(  # seq_len, frames, h, w
                video_emb,
                text_emb,
                "b t h w head c, b s head c -> s t h w",
            )
            ts_attn_maps.append(attn_maps)
        ts_attn_maps = np.stack(ts_attn_maps)  # n_blks, seq_len, t, h, w
        all_attn_maps.append(ts_attn_maps)
    all_attn_maps = np.stack(all_attn_maps)  # n_timesteps, n_blks, seq_len, t, h, w
    all_attn_maps = einops.reduce(all_attn_maps, "nt nb s t h w -> s t h w", reduction="mean")  # seq_len, t, h, w
    n_tokens = len(tokens)
    # assert n_tokens == attn_maps.shape[0], f"Token length {len(tokens)} does not match attention map length {attn_maps.shape[0]}"

    # all_attn_maps = torch.from_numpy(all_attn_maps)
    # selected_token_attn = torch.softmax(all_attn_maps[selected_tokens_idxs], 0)
    # all_attn_maps[selected_tokens_idxs] = selected_token_attn
    # attn_maps_softmax = all_attn_maps

    # attn_maps_softmax = torch.softmax(torch.from_numpy(all_attn_maps), 0)
    
    # norm attn map
    token_frame_attn_map = []
    for token_idx, token in enumerate(tokens):
        frame_attn_map = []
        for frame_idx in range(scaled_frames):
            frame_min_val = all_attn_maps[:, frame_idx].min()
            frame_max_val = all_attn_maps[:, frame_idx].max()
            # frame_min_val = all_attn_maps[:, :].min()
            # frame_max_val = all_attn_maps[:, :].max()
            norm_attn_map = ((all_attn_maps[token_idx, frame_idx] - frame_min_val) / (frame_max_val - frame_min_val))
            frame_attn_map.append(norm_attn_map)
        token_frame_attn_map.append(frame_attn_map)
    token_frame_attn_map = np.array(token_frame_attn_map)  # n_tokens, t, h, w

    # softmax
    token_videos = {}
    for token_idx, token in enumerate(tokens):
        token_videos[token] = []
    for frame_idx in range(scaled_frames):
        token_attn = token_frame_attn_map[:, frame_idx]  # n_tokens, h, w
        # token softmax
        # token_attn_softmax = torch.softmax(torch.from_numpy(token_attn), 0).numpy()
        token_attn_softmax = token_attn

        for token_idx, token in enumerate(tokens):
            frame_attn = token_attn_softmax[token_idx]  # h, w
            frame_attn_min_val, frame_attn_max_val = frame_attn.min(), frame_attn.max()
            # frame_attn_min_val, frame_attn_max_val = token_frame_attn_map[token_idx, :].min(), token_frame_attn_map[token_idx, :].max()
            frame_attn = (frame_attn - frame_attn_min_val) / (frame_attn_max_val - frame_attn_min_val)
            colored_attn_map = plt.get_cmap("plasma")(frame_attn)
            colored_attn_map = (colored_attn_map[:, :, :3] * 255).astype(np.uint8)  # (h, w, 3)
            colored_attn_map_pil = PIL.Image.fromarray(colored_attn_map)
            token_videos[token].append(colored_attn_map_pil)
    return token_videos

def build_attn_maps_origin(atten_dict_list, tokens, frames, height, width, 
                    vae_spatial_compression_ratio, vae_temporal_compression_ratio,
                    patch_size_ratio, patch_size_t_ratio):
    # selected_tokens_idxs = []
    # for token_idx, token in enumerate(tokens):
    #     if (nlp(token)[0].pos_ == 'NOUN'):
    #         selected_tokens_idxs.append(token_idx)
    # selected_tokens_idxs = np.array(selected_tokens_idxs)
    scaled_frames = ((frames + 1) // vae_temporal_compression_ratio + 1) // patch_size_t_ratio
    scaled_height = height // vae_spatial_compression_ratio // patch_size_ratio
    scaled_width = width // vae_spatial_compression_ratio // patch_size_ratio
    n_timesteps = len(atten_dict_list)
    n_blks = len(atten_dict_list[0])
    all_attn_maps = []
    for timestep_idx in tqdm(range(n_timesteps)):
        ts_attn_maps = []
        for blk_idx in range(n_blks):
            video_emb = atten_dict_list[timestep_idx][blk_idx]["video_emb"]
            text_emb = atten_dict_list[timestep_idx][blk_idx]["text_emb"]
            video_emb = einops.rearrange(video_emb, "b (t h w) head c -> b t h w head c", t=scaled_frames,
                                          h=scaled_height, w=scaled_width)
            attn_maps = einops.einsum(  # seq_len, frames, h, w
                video_emb,
                text_emb,
                "b t h w head c, b s head c -> s t h w",
            )
            ts_attn_maps.append(attn_maps)
        ts_attn_maps = np.stack(ts_attn_maps)  # n_blks, seq_len, t, h, w
        all_attn_maps.append(ts_attn_maps)
    all_attn_maps = np.stack(all_attn_maps)  # n_timesteps, n_blks, seq_len, t, h, w
    all_attn_maps = einops.reduce(all_attn_maps, "nt nb s t h w -> s t h w", reduction="mean")  # seq_len, t, h, w
    n_tokens = len(tokens)
    # assert n_tokens == attn_maps.shape[0], f"Token length {len(tokens)} does not match attention map length {attn_maps.shape[0]}"

    # all_attn_maps = torch.from_numpy(all_attn_maps)
    # selected_token_attn = torch.softmax(all_attn_maps[selected_tokens_idxs], 0)
    # all_attn_maps[selected_tokens_idxs] = selected_token_attn
    # attn_maps_softmax = all_attn_maps

    # attn_maps_softmax = torch.softmax(torch.from_numpy(all_attn_maps), 0)
    
    # norm attn map
    token_frame_attn_map = []
    for token_idx, token in enumerate(tokens):
        frame_attn_map = []
        for frame_idx in range(scaled_frames):
            frame_min_val = all_attn_maps[:, frame_idx].min()
            frame_max_val = all_attn_maps[:, frame_idx].max()
            # frame_min_val = all_attn_maps[:, :].min()
            # frame_max_val = all_attn_maps[:, :].max()
            norm_attn_map = ((all_attn_maps[token_idx, frame_idx] - frame_min_val) / (frame_max_val - frame_min_val))
            frame_attn_map.append(norm_attn_map)
        token_frame_attn_map.append(frame_attn_map)
    token_frame_attn_map = np.array(token_frame_attn_map)  # n_tokens, t, h, w

    # softmax
    token_videos = {}
    for token_idx, token in enumerate(tokens):
        token_videos[token] = []
    for frame_idx in range(scaled_frames):
        token_attn = token_frame_attn_map[:, frame_idx]  # n_tokens, h, w
        # token softmax
        token_attn_softmax = torch.softmax(torch.from_numpy(token_attn), 0).numpy()
        # token_attn_softmax = token_attn

        for token_idx, token in enumerate(tokens):
            frame_attn = token_attn_softmax[token_idx]  # h, w
            frame_attn_min_val, frame_attn_max_val = frame_attn.min(), frame_attn.max()
            # frame_attn_min_val, frame_attn_max_val = token_frame_attn_map[token_idx, :].min(), token_frame_attn_map[token_idx, :].max()
            frame_attn = (frame_attn - frame_attn_min_val) / (frame_attn_max_val - frame_attn_min_val)
            colored_attn_map = plt.get_cmap("plasma")(frame_attn)
            colored_attn_map = (colored_attn_map[:, :, :3] * 255).astype(np.uint8)  # (h, w, 3)
            colored_attn_map_pil = PIL.Image.fromarray(colored_attn_map)
            token_videos[token].append(colored_attn_map_pil)
    return token_videos

def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)



def inference(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    # Peiyuan: GPU seed will cause A100 and H100 to produce different results .....
    weight_dtype = torch.bfloat16

    if args.transformer_path is not None:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.transformer_path)
    else:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.model_path,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=weight_dtype)

    pipe = HunyuanVideoPipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=weight_dtype)

    pipe.enable_vae_tiling()

    if args.lora_checkpoint_dir is not None:
        print(f"Loading LoRA weights from {args.lora_checkpoint_dir}")
        config_path = os.path.join(args.lora_checkpoint_dir, "lora_config.json")
        with open(config_path, "r") as f:
            lora_config_dict = json.load(f)
        rank = lora_config_dict["lora_params"]["lora_rank"]
        lora_alpha = lora_config_dict["lora_params"]["lora_alpha"]
        lora_scaling = lora_alpha / rank
        pipe.load_lora_weights(args.lora_checkpoint_dir, adapter_name="default")
        pipe.set_adapters(["default"], [lora_scaling])
        print(f"Successfully Loaded LoRA weights from {args.lora_checkpoint_dir}")
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device)
    else:
        pipe.to(device)

    # Generate videos from the input prompt

    if args.prompt_embed_path is not None:
        prompt_embeds = (torch.load(args.prompt_embed_path, map_location="cpu",
                                    weights_only=True).to(device).unsqueeze(0))
        encoder_attention_mask = (torch.load(args.encoder_attention_mask_path, map_location="cpu",
                                             weights_only=True).to(device).unsqueeze(0))
        prompts = None
    elif args.prompt_path is not None:
        prompts = [line.strip() for line in open(args.prompt_path, "r")]
        prompt_embeds = None
        encoder_attention_mask = None
    else:
        prompts = args.prompts
        prompt_embeds = None
        encoder_attention_mask = None

    if prompts is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for prompt in prompts:
                generator = torch.Generator("cpu").manual_seed(args.seed)
                video = pipe(
                    prompt=[prompt],
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).frames
                if nccl_info.global_rank <= 0:
                    os.makedirs(args.output_path, exist_ok=True)
                    suffix = prompt.split(".")[0]
                    export_to_video(
                        video[0],
                        os.path.join(args.output_path, f"{suffix}.mp4"),
                        fps=24,
                    )
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generator = torch.Generator("cpu").manual_seed(args.seed)
            videos = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=encoder_attention_mask,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames

        if nccl_info.global_rank <= 0:
            export_to_video(videos[0], args.output_path + ".mp4", fps=24)


def inference_quantization(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_path

    if args.quantization == "nf4":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_quant_type="nf4",
                                                 llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=torch.bfloat16,
                                                                     quantization_config=quantization_config)
    if args.quantization == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=torch.bfloat16,
                                                                     quantization_config=quantization_config)
    elif not args.quantization:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id,
                                                                     subfolder="transformer/",
                                                                     torch_dtype=torch.bfloat16).to(device)

    print("Max vram for read transformer:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")
    torch.cuda.reset_max_memory_allocated(device)

    if not args.cpu_offload:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        pipe.transformer = transformer
    else:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16, width=args.width, height=args.height)
    torch.cuda.reset_max_memory_allocated(device)
    pipe.scheduler._shift = args.flow_shift
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    print("Max vram for init pipeline:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")
    if args.prompt.endswith('.txt'):
        with open(args.prompt) as f:
            prompts = [line.strip() for line in f.readlines()]
    else:
        prompts = [args.prompt]

    generator = torch.Generator("cpu").manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.cuda.reset_max_memory_allocated(device)


    # load video
    # video = "/home/myw/wuchangli/yk/my_ov/vd/FastVideo/outputs_video/hunyuan_quant/nf4/A boy is playing basketball..mp4"
    # video = "/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/480p/bear"

    video = "/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/VSPW_480p/data/127_-hIVCYO4C90/origin"
    # video = "/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/VSPW_480p/data/1643_T9npC-YHzuE/origin"
    # text_query = ["boy"]
    prompts = video_semantic_dict['127_-hIVCYO4C90']
    # prompts = video_semantic_dict['1643_T9npC-YHzuE']
    prompts = [' '.join(prompts)]
    import pdb
    pdb.set_trace()
    for prompt in prompts:
        # import pdb
        # pdb.set_trace()
        # prompt = "bear"
        # prompt = "wall stair handrail_or_fence window pole floor lamp person"
        # prompt = video_semantic_dict['127_-hIVCYO4C90']
        start_time = time.perf_counter()
        
        # video is dir?
        if os.path.isdir(video):
            # images
            video_list = []
            video_paths = []
            # walk
            for root, dirs, files in os.walk(video):
                for file in files:
                    # check if file is image
                    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        video_paths.append(os.path.join(root, file))
            # sort
            video_paths.sort()
            # load video
            for video_path in video_paths:
                video_list.append(pipe.get_image(video_path)['pixel_values'])
            video_info = torch.cat(video_list, dim=1)  # c, t, h, w
        else:
            video_info = pipe.get_video(video)['pixel_values']
        n_frames = video_info.shape[1]
        h, w = video_info.shape[2], video_info.shape[3]

        # output, atten_dict_list, n_frames = pipe.seg_with_video_input(
        #     # prompt=prompt,
        #     prompt=text_query[0],
        #     height=args.height,
        #     width=args.width,
        #     num_frames=args.num_frames,
        #     num_inference_steps=args.num_inference_steps,
        #     generator=generator,
        #     video="/home/myw/wuchangli/yk/my_ov/video_attn/github_version/assets/A boy is playing basketball..mp4",
        #     text_query=text_query,
        # )

        output, atten_dict_list = pipe.seg_with_video_input_2(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=n_frames,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            video=video,
        )

        # output, atten_dict_list = pipe(
        #     prompt=prompt,
        #     height=args.height,
        #     width=args.width,
        #     num_frames=args.num_frames,
        #     num_inference_steps=args.num_inference_steps,
        #     generator=generator,
        # )

        output = output[0]
        export_to_video(output, os.path.join(args.output_path, f"{prompt[:100]}.mp4"), fps=args.fps)
        # n_frames = args.num_frames
        height, width = args.height, args.width
        vae_spatial_compression_ratio = pipe.vae.spatial_compression_ratio
        vae_temporal_compression_ratio = pipe.vae.temporal_compression_ratio
        patch_size_ratio = pipe.transformer.config.patch_size
        patch_size_t_ratio = pipe.transformer.config.patch_size_t
        # tokens = pipe.tokenizer.tokenize(prompt)
        # tokens = [token[1:] if token.startswith("Ġ") else token for token in tokens]
        tokens = pipe.tokenizer.tokenize(prompt)
        # import pdb
        # pdb.set_trace()
        # import pdb; pdb.set_trace()
        attn_maps = build_attn_maps_origin(atten_dict_list, tokens, 
                                        # n_frames,
                                        n_frames,
                                        height, width, vae_spatial_compression_ratio, vae_temporal_compression_ratio,
                                        patch_size_ratio, patch_size_t_ratio)

        # attn_maps = build_attn_maps(atten_dict_list, tokens, 
        #                             # n_frames,
        #                             n_frames,
        #                             height, width, vae_spatial_compression_ratio, vae_temporal_compression_ratio,
        #                             patch_size_ratio, patch_size_t_ratio)

        # attn_maps = build_attn_maps_v2(atten_dict_list, tokens, 
        #                             # n_frames,
        #                             n_frames,
        #                             height, width, vae_spatial_compression_ratio, vae_temporal_compression_ratio,
        #                             patch_size_ratio, patch_size_t_ratio)
        token_video_dir = os.path.join(args.output_path, f'{prompt}')
        os.makedirs(token_video_dir, exist_ok=True)
        for tok_idx, (key, video_list) in enumerate(attn_maps.items()):
            for idx, frame in enumerate(video_list):
                video_list[idx] = frame.resize((width, height))
            export_to_video(video_list, os.path.join(token_video_dir, f"{tok_idx}_{key}.mp4"), fps=args.fps // 4)
        print("Time:", round(time.perf_counter() - start_time, 2), "seconds")
        print("Max vram for denoise:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the directory containing LoRA checkpoints",
    )
    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument("--neg_prompt", type=str, default=None, help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift", type=int, default=7, help="Flow shift parameter.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "fp8"])
    parser.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument("--prompt-template-video", type=str, default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)

    args = parser.parse_args()
    if args.quantization:
        inference_quantization(args)
    else:
        inference(args)
