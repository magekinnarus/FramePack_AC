import gradio as gr
import numpy as np
import torch
import os
import gc
import traceback
import math
from PIL import Image

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, save_bchw_as_png
from diffusers_helper.utils import resize_and_center_crop, generate_timestamp, save_bcthw_as_png_sequence
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, \
    move_model_to_device_with_memory_preservation, \
    offload_model_from_device_for_memory_preservation, DynamicSwapInstaller, unload_complete_models, \
    load_model_as_complete, fake_diffusers_current_device
from diffusers_helper.thread_utils import AsyncStream, async_run
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.dit_common import *
from diffusers_helper.gradio.progress_bar import make_progress_bar_html, make_progress_bar_css
from bucket_tools import bucket_options, find_nearest_bucket

stream = AsyncStream()

# Patches for numerical stability
torch.nn.LayerNorm.forward = LayerNorm_forward
diffusers.models.normalization.LayerNorm.forward = LayerNorm_forward
diffusers.models.normalization.FP32LayerNorm.forward = FP32LayerNorm_forward
diffusers.models.normalization.RMSNorm.forward = RMSNorm_forward
diffusers.models.normalization.AdaLayerNormContinuous.forward = AdaLayerNormContinuous_forward

high_vram = get_cuda_free_memory_gb() > 20.0

# Pre-load models if in high VRAM mode
if high_vram:
    text_encoder = LlamaModel.from_pretrained("models/text_encoder", torch_dtype=torch.bfloat16).eval()
    tokenizer = LlamaTokenizerFast.from_pretrained("models/text_encoder")
    text_encoder_2 = CLIPTextModel.from_pretrained("models/text_encoder_2", torch_dtype=torch.bfloat16).eval()
    tokenizer_2 = CLIPTokenizer.from_pretrained("models/text_encoder_2")
    image_encoder = SiglipVisionModel.from_pretrained("models/image_encoder", torch_dtype=torch.bfloat16).eval()
    feature_extractor = SiglipImageProcessor.from_pretrained("models/image_encoder")
    vae = AutoencoderKLHunyuanVideo.from_pretrained("models/vae", torch_dtype=torch.bfloat16).eval()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("models/transformer",
                                                                       torch_dtype=torch.bfloat16).eval()
    transformer.enable_gradient_checkpointing()

def get_chunk_boundaries(total_second_length):
    # This function calculates the number of chunks and their boundary descriptions
    # based on the original script's logic and the user's intuitive "inference order" model.
    latent_window_size = 9  # Internal constant
    fps_for_indexing = 29  # Internal constant

    num_chunks = int(max(round((total_second_length * fps_for_indexing) / (latent_window_size * 4)), 1))
    
    if num_chunks <= 1:
        return ["Disabled"]
    
    choices = ["Disabled"] + [f"At start of Generated Chunk {i+1}" for i in range(num_chunks - 1)]
    return choices

def worker(
    stream: AsyncStream, input_image, end_image,
    mid_kf_image_1, mid_kf_pos_1,
    mid_kf_image_2, mid_kf_pos_2,
    mid_kf_image_3, mid_kf_pos_3,
    prompt, n_prompt, seed, total_second_length,
    steps, cfg, gs, rs, gpu_memory_preservation, use_teacache,
    mp4_crf, save_png_sequence
):
    job_id = generate_timestamp()
    output_folder = f'outputs/{job_id}'
    os.makedirs(output_folder, exist_ok=True)
    stream.output_queue.push(('progress', (None, f'Job {job_id} starting...', make_progress_bar_html(0, 'Starting...'))))

    # Internal constants based on the original model's architecture
    latent_window_size = 9
    fps_for_indexing = 29
    
    try:
        # --- Part 1: PREPARATION (The "Parts Bin" and the "Blueprint") ---
        
        # Load models as needed
        if not high_vram:
            unload_complete_models()
            text_encoder = LlamaModel.from_pretrained("models/text_encoder", torch_dtype=torch.bfloat16).eval()
            tokenizer = LlamaTokenizerFast.from_pretrained("models/text_encoder")
            text_encoder_2 = CLIPTextModel.from_pretrained("models/text_encoder_2", torch_dtype=torch.bfloat16).eval()
            tokenizer_2 = CLIPTokenizer.from_pretrained("models/text_encoder_2")
            load_model_as_complete(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, gpu)
        else:
            text_encoder.to(gpu); text_encoder_2.to(gpu)

        # Encode prompts
        prompt_llama, prompt_clip = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        n_prompt_llama, n_prompt_clip = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if not high_vram: unload_complete_models(text_encoder, text_encoder_2)

        # Determine target resolution
        h, w, _ = input_image.shape
        target_h, target_w = find_nearest_bucket(h, w)

        # VAE-Encode all pristine keyframe latents
        if not high_vram:
            unload_complete_models()
            vae = AutoencoderKLHunyuanVideo.from_pretrained("models/vae", torch_dtype=torch.bfloat16).eval()
            load_model_as_complete(vae, gpu)
        else:
            vae.to(gpu)
            
        def process_and_encode(img):
            resized_img = resize_and_center_crop(img, target_w, target_h)
            pt_img = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            return vae_encode(pt_img, vae), resized_img
        
        start_latent, input_image_resized = process_and_encode(input_image)
        Image.fromarray(input_image_resized).save(f'{output_folder}/input_start.png')

        end_latent, end_image_resized = (None, None)
        if end_image is not None:
            end_latent, end_image_resized = process_and_encode(end_image)
            Image.fromarray(end_image_resized).save(f'{output_folder}/input_end.png')

        # Calculate Mid-Keyframe Indices using the definitive empirical algorithm
        total_frames = int(total_second_length * fps_for_indexing)
        num_chunks = int(max(round((total_second_length * fps_for_indexing) / (latent_window_size * 4)), 1))
        
        mid_keyframe_data = []
        mid_keyframe_inputs = [(mid_kf_image_1, mid_kf_pos_1), (mid_kf_image_2, mid_kf_pos_2), (mid_kf_image_3, mid_kf_pos_3)]

        if num_chunks > 1:
            # Base index for the start of the first generated chunk (chronologically, near the end of the video)
            base_index_T = 40 + (num_chunks - 2) * 36

            for i, (img, pos_str) in enumerate(mid_keyframe_inputs):
                if img is not None and pos_str != "Disabled":
                    # User selects "Chunk 1", "Chunk 2", etc. in inference order.
                    target_inference_chunk_num = int(pos_str.split(" ")[-1])

                    # Formula derived from your empirical analysis
                    frame_index = base_index_T + 1 - (36 * (target_inference_chunk_num - 1))
                    frame_index = max(0, min(frame_index, total_frames - 1))

                    mid_latent, mid_img_resized = process_and_encode(img)
                    Image.fromarray(mid_img_resized).save(f'{output_folder}/input_mid_{i+1}.png')
                    mid_keyframe_data.append({'idx': frame_index, 'latent': mid_latent})

        # Global Image Conditioning
        if not high_vram:
            unload_complete_models()
            image_encoder = SiglipVisionModel.from_pretrained("models/image_encoder", torch_dtype=torch.bfloat16).eval()
            feature_extractor = SiglipImageProcessor.from_pretrained("models/image_encoder")
            load_model_as_complete(image_encoder, gpu)
        else:
            image_encoder.to(gpu)
            
        image_embeds = hf_clip_vision_encode(input_image_resized, feature_extractor, image_encoder).image_embeds
        if end_image_resized is not None:
            end_image_embeds = hf_clip_vision_encode(end_image_resized, feature_extractor, image_encoder).image_embeds
            image_embeds = (image_embeds + end_image_embeds) / 2.0

        if not high_vram: unload_complete_models(image_encoder)

        # Create the Master Blueprint (`clean_latents` canvas)
        latent_h, latent_w = target_h // 8, target_w // 8
        clean_latents = torch.zeros([1, 4, total_frames, latent_h, latent_w], dtype=torch.bfloat16, device=cpu)
        clean_latents_indices = torch.zeros([total_frames], dtype=torch.long, device=cpu)

        # Imprint ALL keyframes onto the blueprint for guidance
        clean_latents[0, :, 0] = start_latent[0, :, 0]; clean_latents_indices[0] = 1
        if end_latent is not None:
            clean_latents[0, :, -1] = end_latent[0, :, 0]; clean_latents_indices[-1] = 1
        for kf in mid_keyframe_data:
            clean_latents[0, :, kf['idx']] = kf['latent'][0, :, 0]; clean_latents_indices[kf['idx']] = 1

        # --- Part 2: THE REVERSE ASSEMBLY LINE ---
        if not high_vram:
            unload_complete_models(vae)
            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("models/transformer", torch_dtype=torch.bfloat16).eval()
            transformer.enable_gradient_checkpointing()
            DynamicSwapInstaller.install_model(transformer, device=gpu)
            fake_diffusers_current_device(transformer, gpu)
        else:
            transformer.to(gpu)
        
        transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)
        history_latents = None
        
        latent_paddings = list(range(total_frames, 0, -latent_window_size))
        latent_paddings = [p - latent_window_size for p in latent_paddings]
        if not latent_paddings or latent_paddings[0] != 0: latent_paddings.insert(0,0)
        latent_paddings = sorted(list(set(latent_paddings)))

        # Loop in chronological order, but assemble in reverse
        for i, latent_padding_start in enumerate(reversed(latent_paddings)):
            if stream.input_queue.pop() == 'end': raise Exception("Process ended by user.")
            
            is_first_chunk_generated = (i == 0) # End of video
            is_last_chunk_generated = (i == len(latent_paddings) - 1) # Start of video

            stream.output_queue.push(('progress', (None, f'Generating Section {i + 1}/{len(latent_paddings)} (Reverse Order)...',
                                                   make_progress_bar_html(int(100 * (i / len(latent_paddings))),'Generating...'))))

            start_t = latent_padding_start
            end_t = min(start_t + latent_window_size + 3, total_frames)
            real_window_size = end_t - start_t
            
            if real_window_size <= 0: continue

            noise = torch.randn([1, 4, real_window_size, latent_h, latent_w], generator=torch.manual_seed(seed), device=gpu, dtype=torch.bfloat16)
            indices_slice = clean_latents_indices[start_t:end_t].to(gpu)
            clean_in_slice = clean_latents[:, :, start_t:end_t].to(gpu)
            
            generated_chunk = sample_hunyuan(transformer, num_inference_steps=steps, frames=real_window_size,
                                             height=target_h, width=target_w, real_guidance_scale=gs,
                                             guidance_rescale=rs, distilled_guidance_scale=cfg,
                                             prompt_embeds=prompt_llama, prompt_poolers=prompt_clip,
                                             negative_prompt_embeds=n_prompt_llama, negative_prompt_poolers=n_prompt_clip,
                                             image_embeddings=image_embeds,
                                             clean_latents={'1x': (clean_in_slice, indices_slice)},
                                             initial_latent=noise, dtype=torch.bfloat16, device=gpu)
            generated_chunk = generated_chunk.to(cpu)
            
            # Assemble the timeline. Prepend the new chunk to the history.
            if history_latents is None:
                history_latents = generated_chunk
            else:
                history_latents, _ = soft_append_bcthw(generated_chunk, history_latents, 3)

        # Symmetrical Quality Guarantee for Start and End Frames
        if end_latent is not None:
            history_latents, _ = soft_append_bcthw(history_latents, end_latent.to(cpu), 3)

        history_latents, _ = soft_append_bcthw(start_latent.to(cpu), history_latents, 3)

        # --- Part 3: FINALIZATION ---
        if not high_vram: load_model_as_complete(vae, gpu)
        final_pixels = vae_decode(history_latents, vae)
        if not high_vram: unload_complete_models(vae)
        
        output_filename = f'{output_folder}/final_video.mp4'
        save_bcthw_as_mp4(final_pixels, output_filename, fps=fps_for_indexing, crf=mp4_crf)
        stream.output_queue.push(('file', output_filename))

        if save_png_sequence:
            png_output_folder = f'{output_folder}/frames'
            stream.output_queue.push(('progress', (None, f'Saving PNG sequence to {png_output_folder}', make_progress_bar_html(100, 'Saving frames...'))))
            save_bcthw_as_png_sequence(final_pixels, png_output_folder)

    except Exception as e:
        print(f"Error in worker: {e}")
        traceback.print_exc()
    finally:
        unload_complete_models()
        gc.collect()
        torch.cuda.empty_cache()
        stream.output_queue.push(('end', None))
    return


def process(
    input_image, end_image, 
    mid_kf_image_1, mid_kf_pos_1,
    mid_kf_image_2, mid_kf_pos_2,
    mid_kf_image_3, mid_kf_pos_3,
    prompt, n_prompt, seed, total_second_length, steps, cfg, gs, rs, 
    gpu_memory_preservation, use_teacache, mp4_crf, save_png_sequence
):
    global stream
    assert input_image is not None, "You must upload a start image."

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    stream = AsyncStream()
    async_run(
        worker, 
        stream, input_image, end_image, 
        mid_kf_image_1, mid_kf_pos_1, mid_kf_image_2, mid_kf_pos_2, mid_kf_image_3, mid_kf_pos_3,
        prompt, n_prompt, seed, total_second_length, steps, cfg, gs, rs, 
        gpu_memory_preservation, use_teacache, mp4_crf, save_png_sequence
    )

    output_filename = None
    while True:
        try:
            flag, data = stream.output_queue.next()
            if flag == 'file':
                output_filename = data
                yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
            if flag == 'progress':
                preview, desc, html = data
                if preview is not None:
                     yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
                else:
                     yield gr.update(), gr.update(), desc, html, gr.update(interactive=False), gr.update(interactive=True)
            if flag == 'end':
                yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
                break
        except Exception as e:
            print(f"Error in UI update loop: {e}"); break

def end_process():
    stream.input_queue.push('end')

with gr.Blocks(css=make_progress_bar_css()) as block:
    gr.Markdown("# FramePack Artist Control")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(sources='upload', type="numpy", label="Start Frame")
                end_image = gr.Image(sources='upload', type="numpy", label="End Frame (Optional)")
            
            prompt = gr.Textbox(label="Prompt", value="A beautiful girl")
            
            with gr.Accordion("Mid-Keyframes (Optional)", open=False):
                gr.Markdown("Place keyframes at the start of a generation chunk (in the order they are generated).")
                mid_kf_pos_1 = gr.Dropdown(label="Mid-Keyframe 1 Position", choices=["Disabled"], value="Disabled")
                mid_kf_image_1 = gr.Image(sources='upload', type="numpy", label="Mid-Keyframe 1 Image")
                mid_kf_pos_2 = gr.Dropdown(label="Mid-Keyframe 2 Position", choices=["Disabled"], value="Disabled")
                mid_kf_image_2 = gr.Image(sources='upload', type="numpy", label="Mid-Keyframe 2 Image")
                mid_kf_pos_3 = gr.Dropdown(label="Mid-Keyframe 3 Position", choices=["Disabled"], value="Disabled")
                mid_kf_image_3 = gr.Image(sources='upload', type="numpy", label="Mid-Keyframe 3 Image")

            n_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, low quality")

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Accordion("Advanced Settings", open=True):
                save_png_sequence = gr.Checkbox(label="Save final PNG sequence", value=True)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1.0, maximum=10.0, value=5.0, step=0.1)
                
                def update_dropdown_choices(length):
                    choices = get_chunk_boundaries(length)
                    return gr.update(choices=choices, value="Disabled"), gr.update(choices=choices, value="Disabled"), gr.update(choices=choices, value="Disabled")

                total_second_length.change(update_dropdown_choices, inputs=[total_second_length], outputs=[mid_kf_pos_1, mid_kf_pos_2, mid_kf_pos_3])

                seed = gr.Number(label="Seed", value=123)
                steps = gr.Slider(label="Steps", minimum=10, maximum=50, value=25, step=1)
                gs = gr.Slider(label="Guidance Scale (Text)", minimum=1.0, maximum=15.0, value=6.0, step=0.1)
                cfg = gr.Slider(label="CFG", minimum=1.0, maximum=15.0, value=1.5, step=0.1)
                rs = gr.Slider(label="Rescale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                gpu_memory_preservation = gr.Slider(label="GPU Memory Preservation (GB)", minimum=4.0, maximum=32.0, value=16.0, step=1.0)
                mp4_crf = gr.Slider(label="MP4 CRF (Quality)", minimum=0, maximum=51, value=23, step=1)
                use_teacache = gr.Checkbox(label="Use TeaCache", value=True)

        with gr.Column():
            result_video = gr.Video(label="Result", height=512)
            preview_image = gr.Image(label="Preview", visible=False, height=256)
            progress_desc = gr.Markdown("")
            progress_bar = gr.HTML("")

    ips = [
        input_image, end_image, 
        mid_kf_image_1, mid_kf_pos_1, mid_kf_image_2, mid_kf_pos_2, mid_kf_image_3, mid_kf_pos_3,
        prompt, n_prompt, seed, total_second_length, steps, cfg, gs, rs, 
        gpu_memory_preservation, use_teacache, mp4_crf, save_png_sequence
    ]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process, inputs=[], outputs=[])

block.launch()