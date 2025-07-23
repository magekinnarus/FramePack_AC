from diffusers_helper.hf_login import login

import os

# os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

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

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16, local_files_only=True).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16, local_files_only=True).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer', local_files_only=True)
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2', local_files_only=True)
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16, local_files_only=True).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor', local_files_only=True)
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16, local_files_only=True).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16, local_files_only=True).cpu()

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
        # --- Initialization from Keyframe 1 ---
        stream.output_queue.push(('progress_update', (None, 'Initializing from Keyframe 1...', make_progress_bar_html(0, 'Init...'), 0.0)))
        
        # Determine consistent dimensions from the first keyframe
        H, W, C = keyframe_sequence[0].shape
        height, width = find_nearest_bucket(H, W, resolution=640)

        if not high_vram: load_model_as_complete(vae, target_device=gpu)
        
        # Process, encode, and decode the very first frame to start the history
        kf1_np = resize_and_center_crop(keyframe_sequence[0], target_width=width, target_height=height)
        kf1_pt = torch.from_numpy(kf1_np).float() / 127.5 - 1
        kf1_pt = kf1_pt.permute(2, 0, 1)[None, :, None]
        
        initial_latent = vae_encode(kf1_pt, vae)
        history_pixels = vae_decode(initial_latent, vae).cpu()
        
        # The initial latent has a single time dimension. This becomes our master latent history.
        history_latents = initial_latent.cpu()
        
        if not high_vram: unload_complete_models(vae)

        # --- Main Orchestration Loop ---
        for i in range(num_runs):
            # Pre-run check for stop signal
            if stream.input_queue.top() == 'stop_all_runs':
                stream.input_queue.pop() # Consume the signal
                stream.output_queue.push(('progress_update', (None, 'All runs stopped by user.', make_progress_bar_html(100, 'Stopped'), 1.0)))
                break

            run_index = i + 1
            
            # --- Prepare Context from History ---
            history_len = history_latents.shape[2]
            
            # Context for 2x downsampling (needs 2 frames)
            if history_len < 2:
                padding_shape = list(history_latents.shape)
                padding_shape[2] = 2 - history_len
                padding = torch.zeros(padding_shape, dtype=history_latents.dtype, device=history_latents.device)
                context_2x_latents = torch.cat([padding, history_latents], dim=2)
            else:
                context_2x_latents = history_latents[:, :, -2:]

            # Context for 4x downsampling (needs 16 frames)
            if history_len < 16:
                padding_shape = list(history_latents.shape)
                padding_shape[2] = 16 - history_len
                padding = torch.zeros(padding_shape, dtype=history_latents.dtype, device=history_latents.device)
                context_4x_latents = torch.cat([padding, history_latents], dim=2)
            else:
                context_4x_latents = history_latents[:, :, -16:]

            # --- Per-Run Setup ---
            start_image_for_run = keyframe_sequence[i]
            is_open_ended_run = (i == num_runs - 1) and perform_open_ended
            end_image_for_run = None if is_open_ended_run else keyframe_sequence[i + 1]
            prompt_for_run = processed_prompts[i]
            
            length_for_this_run = parsed_lengths[i] if i < len(parsed_lengths) else parsed_lengths[-1]
            num_frames_to_generate = int(length_for_this_run * 30)

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

                # Image Processing
                start_image_np = resize_and_center_crop(start_image_for_run, target_width=width, target_height=height)
                start_image_pt = torch.from_numpy(start_image_np).float() / 127.5 - 1
                start_image_pt = start_image_pt.permute(2, 0, 1)[None, :, None]

                # VAE Encoding
                if not high_vram: load_model_as_complete(vae, target_device=gpu)
                start_latent = vae_encode(start_image_pt, vae)
                
                if end_image_for_run:
                    end_image_np = resize_and_center_crop(end_image_for_run, target_width=width, target_height=height)
                    end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                    end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]
                    end_latent = vae_encode(end_image_pt, vae)
                else:
                    end_latent = None

                # CLIP Vision Encoding
                if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
                image_encoder_output = hf_clip_vision_encode(start_image_np, feature_extractor, image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
                if end_image_for_run:
                    end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
                    image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_output.last_hidden_state) / 2

                # Data types and device placement
                llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state = [t.to(transformer.dtype) for t in [llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state]]

                # Sampling
                stream.output_queue.push(('progress_update', (None, f'Run {run_index}/{num_runs}: Sampling (Seed: {current_seed_for_run}, Length: {length_for_this_run}s)...', make_progress_bar_html(0, 'Sampling...'), i / num_runs)))
                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)

                rnd = torch.Generator("cpu").manual_seed(current_seed_for_run)

                # Callback for progress and control
                def callback(d):
                    signal = stream.input_queue.top() # Peek, don't pop
                    if signal in ['skip_current_run', 'stop_all_runs']:
                        raise KeyboardInterrupt()

                    preview = vae_decode_fake(d['denoised'])
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Run {run_index}/{num_runs}. Total video length: {(history_pixels.shape[2] / 30.0):.2f}s'
                    run_progress = (i + (current_step / steps)) / num_runs
                    stream.output_queue.push(('progress_update', (preview, desc, make_progress_bar_html(percentage, hint), run_progress)))

                # Setup latent indices and clean latents for this run
                indices = torch.arange(0, sum([1, 0, num_frames_to_generate, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, 0, num_frames_to_generate, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                
                # Anchor latents for the current run
                clean_latents_for_run = torch.cat(
                    [start_latent, end_latent if end_latent is not None else torch.zeros_like(start_latent)], 
                    dim=2
                )

                generated_latents = sample_hunyuan(
                    transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames_to_generate,
                    real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                    num_inference_steps=steps, generator=rnd,
                    prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu, dtype=torch.bfloat16, image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices, 
                    clean_latents=clean_latents_for_run, clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=context_2x_latents, clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=context_4x_latents, clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                # --- VAE Decode and History Update ---
                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, gpu, 8)
                    load_model_as_complete(vae, gpu)

                current_pixels = vae_decode(generated_latents, vae).cpu()
                
                # The last frame of history IS the first frame of the new segment. Blend over it.
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlap=1)
                
                # Update the master latent history, excluding the duplicated first frame of the new segment
                history_latents = torch.cat([history_latents, generated_latents.cpu()[:, :, 1:]], dim=2)
                
            except KeyboardInterrupt:
                signal = stream.input_queue.pop() # Consume the signal now
                if signal == 'skip_current_run':
                    stream.output_queue.push(('progress_update', (None, f'Run {run_index}/{num_runs} skipped.', make_progress_bar_html(100, 'Skipped'), (run_index) / num_runs)))
                    continue
                elif signal == 'stop_all_runs':
                    stream.output_queue.push(('progress_update', (None, 'All runs stopped.', make_progress_bar_html(100, 'Stopped'), 1.0)))
                    break
            
            # Save intermediate cumulative video
            output_filename = os.path.join(session_folder, f'run_{run_index}_cumulative.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            stream.output_queue.push(('file', output_filename))

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
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, overall_progress, gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)
        elif flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
            break

def skip_current_run():
    # Push only if there isn't already a command, to prevent queue buildup
    if stream.input_queue.top() is None:
        stream.input_queue.push('skip_current_run')

def stop_all_runs():
    # Push only if there isn't already a command
    if stream.input_queue.top() is None:
        stream.input_queue.push('stop_all_runs')

def update_ui_based_on_inputs(num_additional_keyframes, is_open_ended, k1, k2, k3, k4, k5):
    num_additional = int(num_additional_keyframes)
    
    # Keyframe visibility
    kf3_vis = gr.update(visible=num_additional >= 1)
    kf4_vis = gr.update(visible=num_additional >= 2)
    kf5_vis = gr.update(visible=num_additional >= 3)
    
    # Calculate number of runs based on *actual* inputs
    all_kf_inputs = [k1, k2, k3, k4, k5]
    num_valid_keyframes = sum(1 for kf in all_kf_inputs[:2+num_additional] if kf is not None)
    
    num_runs = max(0, num_valid_keyframes - 1)
    if is_open_ended and num_valid_keyframes > 0:
        num_runs += 1
        
    # Prompt visibility
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
                
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
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
            overall_progress = gr.Progress()
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