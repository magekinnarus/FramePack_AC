import os

DOWNLOAD_ROOT = '/content/downloaded_models'

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp, save_bcthw_as_png_sequence
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags
print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load models from local paths
text_encoder = LlamaModel.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'hunyuanvideo_text_encoder'), torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'hunyuanvideo_text_encoder_2'), torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'hunyuanvideo_tokenizer'))
tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'hunyuanvideo_tokenizer_2'))
vae = AutoencoderKLHunyuanVideo.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'hunyuanvideo_vae'), torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'flux_feature_extractor'))
image_encoder = SiglipVisionModel.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'flux_image_encoder'), torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(os.path.join(DOWNLOAD_ROOT, 'framepack_transformer'), torch_dtype=torch.bfloat16).cpu()


vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(all_keyframes, all_prompts, n_prompt, perform_open_ended, seed, increment_seed, lengths_str, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, save_as_png_sequence):
    session_id = generate_timestamp()
    session_folder = os.path.join(outputs_folder, session_id)
    os.makedirs(session_folder, exist_ok=True)

    # --- Input Validation and Processing ---
    keyframe_sequence = [k for k in all_keyframes if k is not None]
    
    if not keyframe_sequence:
        stream.output_queue.push(('progress_update', (None, 'Error: Keyframe 1 is required.', make_progress_bar_html(0, 'Error'), 0.0)))
        stream.output_queue.push(('end', None))
        return

    num_keyframes = len(keyframe_sequence)
    num_runs = max(0, num_keyframes - 1)
    if perform_open_ended:
        num_runs += 1

    if num_runs == 0:
        stream.output_queue.push(('progress_update', (None, 'Not enough keyframes for a run. At least 2 keyframes, or 1 with "open-ended" checked, are needed.', make_progress_bar_html(0, 'Error'), 0.0)))
        stream.output_queue.push(('end', None))
        return

    # Process prompts with fallback logic
    processed_prompts = []
    last_good_prompt = ""
    for i in range(num_runs):
        current_prompt = all_prompts[i] if i < len(all_prompts) else ""
        if current_prompt and current_prompt.strip():
            last_good_prompt = current_prompt
            processed_prompts.append(current_prompt)
        else:
            processed_prompts.append(last_good_prompt)
            
    # Process lengths per run with fallback
    parsed_lengths = []
    try:
        str_values = lengths_str.split(',')
        for v in str_values:
            if v.strip():
                parsed_lengths.append(float(v.strip()))
    except ValueError:
        print(f"Warning: Invalid input for lengths '{lengths_str}'. Falling back to default.")
        parsed_lengths = []

    if not parsed_lengths:
        parsed_lengths = [4.0]

    try:
        # --- Initialization ---
        stream.output_queue.push(('progress_update', (None, 'Initializing...', make_progress_bar_html(0, 'Init...'), 0.0)))
        H, W, C = keyframe_sequence[0].shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        history_pixels = None

        # --- Main Orchestration Loop (Per Storyboard Run) ---
        for i in range(num_runs):
            if stream.input_queue.top() == 'stop_all_runs':
                stream.input_queue.pop()
                stream.output_queue.push(('progress_update', (None, 'All runs stopped by user.', make_progress_bar_html(100, 'Stopped'), 1.0)))
                break

            run_index = i + 1
            
            # --- Per-Run Setup ---
            start_image_for_run = keyframe_sequence[i]
            is_open_ended_run = (i == num_runs - 1) and perform_open_ended
            end_image_for_run = None if is_open_ended_run else keyframe_sequence[i + 1]
            has_end_image = end_image_for_run is not None
            prompt_for_run = processed_prompts[i]
            
            length_for_this_run = parsed_lengths[i] if i < len(parsed_lengths) else parsed_lengths[-1]
            current_seed_for_run = seed + i if increment_seed else seed
            
            # --- Per-Run Execution Logic ---
            try:
                # Text encoding
                stream.output_queue.push(('progress_update', (None, f'Run {run_index}/{num_runs}: Text encoding...', make_progress_bar_html(0, 'Text encoding...'), i / num_runs)))
                if not high_vram:
                    fake_diffusers_current_device(text_encoder, gpu)
                    load_model_as_complete(text_encoder_2, target_device=gpu)

                llama_vec, clip_l_pooler = encode_prompt_conds(prompt_for_run, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                llama_vec_n, clip_l_pooler_n = (torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)) if cfg == 1 else encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

                # VAE and CLIP Vision Encoding for start/end frames of this run
                if not high_vram: load_model_as_complete(vae, target_device=gpu)
                start_image_np = resize_and_center_crop(start_image_for_run, target_width=width, target_height=height)
                start_image_pt = (torch.from_numpy(start_image_np).float() / 127.5 - 1).permute(2, 0, 1)[None, :, None]
                start_latent = vae_encode(start_image_pt, vae)
                
                if has_end_image:
                    end_image_np = resize_and_center_crop(end_image_for_run, target_width=width, target_height=height)
                    end_image_pt = (torch.from_numpy(end_image_np).float() / 127.5 - 1).permute(2, 0, 1)[None, :, None]
                    end_latent = vae_encode(end_image_pt, vae)

                if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
                image_encoder_output = hf_clip_vision_encode(start_image_np, feature_extractor, image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
                if has_end_image:
                    end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
                    image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_output.last_hidden_state) / 2

                llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state = [t.to(transformer.dtype) for t in [llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state]]

                # --- Memory-Safe Chunking Loop (Inner Loop) ---
                total_latent_sections = (length_for_this_run * 30) / (latent_window_size * 4)
                total_latent_sections = int(max(round(total_latent_sections), 1))
                
                latent_paddings = list(reversed(range(total_latent_sections)))
                if total_latent_sections > 4:
                    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

                rnd = torch.Generator("cpu").manual_seed(current_seed_for_run)
                chunk_history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
                total_generated_latent_frames_in_run = 0

                for chunk_idx, latent_padding in enumerate(latent_paddings):
                    is_last_section = latent_padding == 0
                    is_first_section = latent_padding == latent_paddings[0]
                    latent_padding_size = latent_padding * latent_window_size

                    stream.output_queue.push(('progress_update', (None, f'Run {run_index}/{num_runs}, Chunk {chunk_idx+1}/{len(latent_paddings)}... (Seed: {current_seed_for_run})', make_progress_bar_html(0, 'Sampling chunk...'), (i + chunk_idx/len(latent_paddings)) / num_runs )))
                    
                    indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                    clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                    clean_latents_pre = start_latent.to(chunk_history_latents)
                    clean_latents_post, clean_latents_2x, clean_latents_4x = chunk_history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                    
                    if has_end_image and is_first_section:
                        clean_latents_post = end_latent.to(chunk_history_latents)
                    
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                    
                    if not high_vram:
                        unload_complete_models()
                        move_model_to_device_with_memory_preservation(transformer, gpu, gpu_memory_preservation)
                    
                    transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)

                    def callback(d):
                        signal = stream.input_queue.top()
                        if signal in ['skip_current_run', 'stop_all_runs']:
                            raise KeyboardInterrupt()

                        preview = vae_decode_fake(d['denoised'])
                        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                        current_step = d['i'] + 1
                        percentage = int(100.0 * current_step / steps)
                        hint = f'Sampling {current_step}/{steps}'
                        desc = f'Run {run_index}/{num_runs}, Chunk {chunk_idx+1}/{len(latent_paddings)}. Total video length: {((history_pixels.shape[2] if history_pixels is not None else 0) / 30.0):.2f}s'
                        run_progress = (i + (chunk_idx + current_step/steps) / len(latent_paddings)) / num_runs
                        stream.output_queue.push(('progress_update', (preview, desc, make_progress_bar_html(percentage, hint), run_progress)))

                    num_frames_in_chunk = latent_window_size * 4 - 3
                    
                    generated_latents_chunk = sample_hunyuan(
                        transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames_in_chunk,
                        real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                        num_inference_steps=steps, generator=rnd,
                        prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu, dtype=torch.bfloat16, image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )

                    if is_last_section:
                        generated_latents_chunk = torch.cat([start_latent.to(generated_latents_chunk), generated_latents_chunk], dim=2)
                    
                    total_generated_latent_frames_in_run += generated_latents_chunk.shape[2]
                    chunk_history_latents = torch.cat([generated_latents_chunk.to(chunk_history_latents), chunk_history_latents], dim=2)

                    if not high_vram:
                        offload_model_from_device_for_memory_preservation(transformer, gpu, 8)
                        load_model_as_complete(vae, gpu)

                    real_chunk_history_latents = chunk_history_latents[:, :, :total_generated_latent_frames_in_run]

                    if history_pixels is None:
                        history_pixels = vae_decode(real_chunk_history_latents, vae).cpu()
                    else:
                        section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                        overlapped_frames = latent_window_size * 4 - 3
                        
                        current_pixels = vae_decode(real_chunk_history_latents[:, :, :section_latent_frames], vae).cpu()
                        history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                    if not high_vram: unload_complete_models()

                    output_filename = os.path.join(session_folder, f'run_{run_index}_cumulative.mp4')
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
                    stream.output_queue.push(('file', output_filename))
                    
                    if is_last_section:
                        break # Exit the inner chunking loop for this run
            
            except KeyboardInterrupt:
                signal = stream.input_queue.pop()
                if signal == 'skip_current_run':
                    stream.output_queue.push(('progress_update', (None, f'Run {run_index}/{num_runs} skipped.', make_progress_bar_html(100, 'Skipped'), (run_index) / num_runs)))
                    # If skipping the first run, we need a placeholder pixel history
                    if history_pixels is None:
                        if not high_vram: load_model_as_complete(vae, target_device=gpu)
                        history_pixels = vae_decode(start_latent, vae).cpu()
                        if not high_vram: unload_complete_models(vae)
                    continue
                elif signal == 'stop_all_runs':
                    stream.output_queue.push(('progress_update', (None, 'All runs stopped.', make_progress_bar_html(100, 'Stopped'), 1.0)))
                    break

        # --- Final Output ---
        if history_pixels is not None and save_as_png_sequence:
            stream.output_queue.push(('progress_update', (None, f'Saving final PNG sequence...', make_progress_bar_html(100, 'Saving...'), 1.0)))
            png_sequence_dir = os.path.join(session_folder, 'png_sequence')
            save_bcthw_as_png_sequence(history_pixels, png_sequence_dir)

    except Exception:
        traceback.print_exc()
    finally:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    
    stream.output_queue.push(('end', None))


def process(k1, k2, k3, k4, k5, p1, p2, p3, p4, p5, n_prompt, open_ended, seed, increment_seed, lengths_str, lws, steps, cfg, gs, rs, mem, teacache, crf, png):
    global stream
    assert k1 is not None, 'Keyframe 1 is required!'
    all_keyframes = [k1, k2, k3, k4, k5]
    all_prompts = [p1, p2, p3, p4, p5]
    
    yield None, None, '', '', 0.0, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

    stream = AsyncStream()
    async_run(worker, all_keyframes, all_prompts, n_prompt, open_ended, seed, increment_seed, lengths_str, lws, steps, cfg, gs, rs, mem, teacache, crf, png)

    output_filename = None
    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)
        elif flag == 'progress_update':
            preview, desc, html, overall_progress = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, overall_progress*100, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)
        elif flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', 0.0, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
            break

def skip_current_run():
    if stream.input_queue.top() is None:
        stream.input_queue.push('skip_current_run')

def stop_all_runs():
    if stream.input_queue.top() is None:
        stream.input_queue.push('stop_all_runs')

def update_ui_based_on_inputs(num_additional_keyframes, is_open_ended, k1, k2, k3, k4, k5):
    num_additional = int(num_additional_keyframes)
    
    kf3_vis = gr.update(visible=num_additional >= 1)
    kf4_vis = gr.update(visible=num_additional >= 2)
    kf5_vis = gr.update(visible=num_additional >= 3)
    
    all_kf_inputs = [k1, k2, k3, k4, k5]
    num_valid_keyframes = sum(1 for kf in all_kf_inputs[:2+num_additional] if kf is not None)
    
    num_runs = max(0, num_valid_keyframes - 1)
    if is_open_ended and num_valid_keyframes > 0:
        num_runs += 1
        
    p2_vis = gr.update(visible=num_runs >= 2)
    p3_vis = gr.update(visible=num_runs >= 3)
    p4_vis = gr.update(visible=num_runs >= 4)
    p5_vis = gr.update(visible=num_runs >= 5)

    return f"{num_runs}", kf3_vis, kf4_vis, kf5_vis, p2_vis, p3_vis, p4_vis, p5_vis


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack Storyboard')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Storyboard Inputs", open=True):
                with gr.Row():
                    k1_image = gr.Image(sources='upload', type="numpy", label="Keyframe 1 (Required)", height=200)
                    k2_image = gr.Image(sources='upload', type="numpy", label="Keyframe 2", height=200)
                
                num_additional_keyframes_dd = gr.Dropdown(choices=[0, 1, 2, 3], value=0, label="Number of Additional Keyframes", interactive=True)
                
                with gr.Row():
                    k3_image = gr.Image(sources='upload', type="numpy", label="Keyframe 3", height=200, visible=False)
                    k4_image = gr.Image(sources='upload', type="numpy", label="Keyframe 4", height=200, visible=False)
                    k5_image = gr.Image(sources='upload', type="numpy", label="Keyframe 5", height=200, visible=False)
                
                gr.Markdown("---")
                
                p1_prompt = gr.Textbox(label="Prompt for Run 1 (Required)", lines=2)
                p2_prompt = gr.Textbox(label="Prompt for Run 2", lines=2, visible=False)
                p3_prompt = gr.Textbox(label="Prompt for Run 3", lines=2, visible=False)
                p4_prompt = gr.Textbox(label="Prompt for Run 4", lines=2, visible=False)
                p5_prompt = gr.Textbox(label="Prompt for Run 5", lines=2, visible=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation", variant="primary")
                skip_button = gr.Button(value="Skip Current Run", interactive=False)
                stop_button = gr.Button(value="Stop All Runs", interactive=False)

            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    open_ended_checkbox = gr.Checkbox(label="Perform final open-ended run", value=False, interactive=True)
                    total_runs_display = gr.Textbox(label="Total Runs to Execute", value="0", interactive=False)
                
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                save_as_png_sequence = gr.Checkbox(label="Save as PNG Sequence", value=False, info="Saves the final video as a PNG sequence in a subfolder inside 'outputs'.")
                
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=31337, precision=0)
                increment_seed_checkbox = gr.Checkbox(label="Increment seed per run", value=False, info="If checked, each run uses 'seed + run_number'. If unchecked (default), all runs use the same seed for reproducibility.")
                
                lengths_per_run_str = gr.Textbox(label="Video Lengths Per Run (seconds)", value="4", info="Comma-separated list (e.g., '4, 2, 5'). If fewer values than runs, the last value is used for all remaining runs.")
                
                latent_window_size = gr.Slider(label="Latent Window Size (Anchors)", minimum=3, maximum=25, value=9, step=2, info="Larger values produce more coherent motion but use more VRAM. ODD NUMBERS ARE STRONGLY RECOMMENDED.")
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1, info="Larger value causes slower speed but helps avoid OOM.")
                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed.")
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

        with gr.Column(scale=3):
            result_video = gr.Video(label="Cumulative Video", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown("---")
            gr.Markdown("### Progress")
            gr.Markdown("Overall Session Progress")
            overall_progress = gr.Slider(label="Overall Session Progress", minimum=0, maximum=100, value=0, step=0.1, interactive=False, elem_classes='no-generating-animation')
            preview_image = gr.Image(label="Current Run Latent Preview", height=200, visible=False)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation', label="Current Run Progress")

    # --- Event Listeners ---
    
    ui_control_inputs = [num_additional_keyframes_dd, open_ended_checkbox, k1_image, k2_image, k3_image, k4_image, k5_image]
    ui_control_outputs = [total_runs_display, k3_image, k4_image, k5_image, p2_prompt, p3_prompt, p4_prompt, p5_prompt]
    
    for component in ui_control_inputs:
        component.change(fn=update_ui_based_on_inputs, inputs=ui_control_inputs, outputs=ui_control_outputs)
        
    process_inputs = [
        k1_image, k2_image, k3_image, k4_image, k5_image,
        p1_prompt, p2_prompt, p3_prompt, p4_prompt, p5_prompt,
        n_prompt, open_ended_checkbox, seed, increment_seed_checkbox, lengths_per_run_str,
        latent_window_size, steps, cfg, gs, rs,
        gpu_memory_preservation, use_teacache, mp4_crf, save_as_png_sequence
    ]
    process_outputs = [
        result_video, preview_image, progress_desc, progress_bar, overall_progress,
        start_button, skip_button, stop_button
    ]
    start_button.click(fn=process, inputs=process_inputs, outputs=process_outputs)
    
    skip_button.click(fn=skip_current_run)
    stop_button.click(fn=stop_all_runs)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)