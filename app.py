"""
Flux Image Generator - Gradio Web Interface

Fast image generation on Apple Silicon and CUDA.
Supports multiple models:
- Z-Image Turbo (quantized/full)
- FLUX.2-klein-4B (int8 quantized)

FLUX.2-klein also supports image-to-image editing!
"""

import os

os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import csv
import torch
import gradio as gr
from PIL import Image
import json
import atexit
import shutil
import tempfile
from datetime import datetime

from anima_aio import (
    ANIMA_DEFAULTS,
    ANIMA_MODEL_CHOICE,
    ANIMA_MODEL_TYPE,
    ANIMA_PRESETS,
    delete_anima_model,
    generate_anima_aio,
    get_anima_preset,
    get_anima_storage_entry,
    is_anima_model_choice,
)

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen"
)


def cleanup_gradio_cache():
    gradio_temp = os.path.join(tempfile.gettempdir(), "gradio")
    if os.path.exists(gradio_temp):
        try:
            shutil.rmtree(gradio_temp)
            print("Cleaned up Gradio cache.")
        except Exception:
            pass


atexit.register(cleanup_gradio_cache)

# Global state
pipe = None
current_device = None
current_model = (
    None  # "zimage-quant", "zimage-full", "flux2-klein-int8", "anima-aio-metal"
)
current_lora_path = None

# Model choices
MODEL_CHOICES = [
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",
    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)",
    "FLUX.2-klein-4B (Int8)",
    "Z-Image Turbo (Quantized - Fast)",
    ANIMA_MODEL_CHOICE,
    "Z-Image Turbo (Full - LoRA support)",
]


def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def load_zimage_pipeline(device="mps", use_full_model=False):
    """Load Z-Image pipeline (quantized or full)."""
    import sdnq  # Required for quantized model
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

    if use_full_model:
        print(f"Loading Z-Image-Turbo (full precision) on {device}...")
        dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        print(f"Loading Z-Image-Turbo UINT4 (quantized) on {device}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def print_memory(label):
    """Print memory usage with label."""
    mem = get_memory_usage()
    print(f"  [MEM] {label}: {mem:.2f} GB")


def load_flux2_klein_pipeline(device="mps"):
    """Load FLUX.2-klein-4B with int8 quantized transformer and text encoder."""
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoConfig
    from optimum.quanto import requantize
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from quantized_flux2 import QuantizedFlux2Transformer2DModel

    print(f"Loading FLUX.2-klein-4B (int8 quantized) on {device}...")
    print_memory("Before loading")

    model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8")

    print("  Loading int8 transformer...")
    qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
    qtransformer.to(device=device, dtype=torch.bfloat16)
    print_memory("After transformer")

    print("  Loading int8 text encoder...")
    config = AutoConfig.from_pretrained(
        f"{model_path}/text_encoder", trust_remote_code=True
    )
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(config)

    with open(f"{model_path}/text_encoder/quanto_qmap.json", "r") as f:
        qmap = json.load(f)
    state_dict = load_file(f"{model_path}/text_encoder/model.safetensors")
    requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
    text_encoder.eval()
    text_encoder.to(device, dtype=torch.bfloat16)
    print_memory("After text encoder")

    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

    print("  Loading VAE and scheduler...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
    )
    print_memory("After VAE/scheduler download")

    pipe.transformer = qtransformer._wrapped
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.to(device)
    print_memory("After pipe.to(device)")

    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-4B ready!")
    return pipe


def load_flux2_klein_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    print(f"Loading FLUX.2-klein-4B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model (SDNQ model missing vocab files)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        subfolder="tokenizer",
        use_fast=False,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    print_memory("After loading")

    pipe.to(device)
    print_memory("After pipe.to(device)")

    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-4B (SDNQ) ready!")
    return pipe


def load_flux2_klein_9b_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    print(f"Loading FLUX.2-klein-9B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        subfolder="tokenizer",
        use_fast=False,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    print_memory("After loading")

    pipe.to(device)
    print_memory("After pipe.to(device)")

    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-9B (SDNQ) ready!")
    return pipe


def load_pipeline(model_choice: str, device: str = "mps"):
    global pipe, current_device, current_model, current_lora_path

    if is_anima_model_choice(model_choice):
        model_type = ANIMA_MODEL_TYPE
    elif "Quantized" in model_choice:
        model_type = "zimage-quant"
    elif "Full" in model_choice:
        model_type = "zimage-full"
    elif "9B" in model_choice and "SDNQ" in model_choice:
        model_type = "flux2-klein-9b-sdnq"
    elif "4bit SDNQ" in model_choice:
        model_type = "flux2-klein-sdnq"
    elif "FLUX" in model_choice:
        model_type = "flux2-klein-int8"
    else:
        model_type = "zimage-quant"

    if model_type == ANIMA_MODEL_TYPE:
        if pipe is not None:
            print(f"Switching from {current_model} to {model_type}...")
            del pipe
            pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        current_device = "metal"
        current_model = model_type
        current_lora_path = None
        print("Using external Anima AIO Metal runner.")
        return None

    if pipe is not None and current_device == device and current_model == model_type:
        return pipe

    if pipe is not None:
        print(f"Switching from {current_model} to {model_type}...")
        del pipe
        current_lora_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if model_type == "flux2-klein-int8":
        pipe = load_flux2_klein_pipeline(device)
    elif model_type == "flux2-klein-sdnq":
        pipe = load_flux2_klein_sdnq_pipeline(device)
    elif model_type == "flux2-klein-9b-sdnq":
        pipe = load_flux2_klein_9b_sdnq_pipeline(device)
    elif model_type == "zimage-full":
        pipe = load_zimage_pipeline(device, use_full_model=True)
    else:
        pipe = load_zimage_pipeline(device, use_full_model=False)

    current_device = device
    current_model = model_type
    print(f"Pipeline loaded on {device}! (Model: {model_type})")
    return pipe


def load_lora(lora_file, lora_strength: float, device: str):
    """Load or update LoRA adapter (Z-Image full model only)."""
    global current_lora_path, pipe

    if current_model != "zimage-full":
        return "LoRA only supported with Z-Image Full model"

    if lora_file is None or lora_file == "":
        if current_lora_path is not None:
            print("Unloading current LoRA...")
            pipe.unload_lora_weights()
            current_lora_path = None
        return "No LoRA loaded"

    lora_path = lora_file if isinstance(lora_file, str) else lora_file.name

    if not os.path.exists(lora_path):
        return f"LoRA file not found: {lora_path}"

    if not lora_path.endswith(".safetensors"):
        return "Please select a .safetensors file"

    if current_lora_path == lora_path:
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        return f"Updated LoRA strength to {lora_strength}"

    if current_lora_path is not None:
        print(f"Unloading previous LoRA: {current_lora_path}")
        pipe.unload_lora_weights()

    try:
        lora_name = os.path.basename(lora_path)
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        current_lora_path = lora_path
        return f"Loaded LoRA: {lora_name} (strength={lora_strength})"
    except Exception as e:
        current_lora_path = None
        return f"Error loading LoRA: {str(e)}"


def update_lora_strength(strength: float):
    """Update the LoRA strength without reloading."""
    global pipe, current_lora_path
    if current_lora_path is not None and pipe is not None:
        try:
            pipe.set_adapters(["default"], adapter_weights=[strength])
            return f"LoRA strength updated to {strength}"
        except Exception as e:
            return f"Error updating strength: {str(e)}"
    return "No LoRA loaded"


# =============================================================================
# Generation Helpers (extracted from generate_image)
# =============================================================================


def _build_generator(device: str, seed: int):
    """Create a PyTorch Generator on the specified device with the given seed."""
    if device == "cuda":
        return torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        return torch.Generator("mps").manual_seed(seed)
    else:
        return torch.Generator().manual_seed(seed)


def _make_inf_params(height, width, prompt=None, image=None):
    """Build kwargs for a pipe() call. Only includes non-None values."""
    params = {"height": int(height), "width": int(width)}
    if prompt is not None:
        params["prompt"] = prompt
    if image is not None:
        params["image"] = image
    return params


def _generate_flux_img2img(
    pipe, prompt, images, height, width, steps, guidance, generator
):
    """Run Flux pipeline in image-to-image mode with tiling disabled."""
    img_w, img_h = int(width), int(height)
    processed_images = []
    for img_data in images[:6]:
        pil_img = img_data[0] if isinstance(img_data, tuple) else img_data
        resized = pil_img.copy().resize((img_w, img_h), Image.LANCZOS)
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        processed_images.append(resized)

    print_memory(f"After resizing {len(processed_images)} image(s)")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
        pipe.vae.disable_tiling()

    ref_input = processed_images[0] if len(processed_images) == 1 else processed_images
    params = _make_inf_params(height=img_h, width=img_w, prompt=prompt, image=ref_input)
    params.update(
        {
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": generator,
        }
    )

    result_image = pipe(**params).images[0]

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    return result_image, f"img2img ({len(processed_images)} ref)"


def _generate_txt2img(pipe, prompt, height, width, steps, guidance, generator):
    """Run any pipeline in text-to-image mode."""
    params = _make_inf_params(height=height, width=width, prompt=prompt)
    params.update(
        {
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": generator,
        }
    )
    return pipe(**params).images[0], "txt2img"


def _cleanup_memory():
    """Free GPU/MPS caches and Python garbage collector."""
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


MODEL_SHORT_NAMES = {
    "zimage-quant": "Z-Image (quant)",
    "zimage-full": "Z-Image (full)",
    "flux2-klein-int8": "FLUX.2-klein-4B (int8)",
    "flux2-klein-sdnq": "FLUX.2-klein-4B (4bit)",
    "flux2-klein-9b-sdnq": "FLUX.2-klein-9B (4bit)",
}


def _format_generation_info(seed, mode, guidance, lora_file, model_short):
    """Build the generation info string shown in the UI."""
    parts = [
        f"Seed: {seed}",
        f"Model: {model_short}",
        f"Mode: {mode}",
        f"Device: {current_device}",
    ]

    if guidance > 0:
        parts.append(f"CFG: {guidance}")

    lora_name = os.path.basename(lora_file) if lora_file else None
    if lora_name:
        strength = getattr(_format_generation_info, "_lora_strength", 1.0) or 1.0
        parts.append(f"LoRA: {lora_name} ({strength})")

    return " | ".join(parts)


def _generate_anima_result(
    prompt, preset_name, steps, guidance, height, width, auto_save, output_dir
):
    """Run the Anima AIO Metal pipeline and build result info."""
    preset = get_anima_preset(preset_name)
    result = generate_anima_aio(
        prompt,
        height=int(height),
        width=int(width),
        steps=int(steps),
        seed=int(0),
        cfg_scale=float(guidance),
        cache_mode=preset["cache_mode"],
        output_dir=output_dir if auto_save else None,
    )

    actual_seed = result.get("seed", 0)
    cfg_info = f" | CFG: {guidance}" if guidance > 0 else ""
    cache_info = f" | Cache: {result.get('cache_mode', preset['cache_mode'])}"
    if result.get("spectrum_skipped"):
        cache_info += f" ({result['spectrum_skipped']} skipped)"
    timing_parts = []
    if result.get("generation_time"):
        timing_parts.append(f" | Gen: {result['generation_time']}")
    if result.get("wall_time"):
        timing_parts.append(f" | Wall: {result['wall_time']}")
    save_info = f" | Saved: {result['path']}" if auto_save else ""

    info = (
        f"Seed: {actual_seed} | Model: Anima Turbo AIO Q4 (Metal) | "
        f"Preset: {preset_name or 'Balanced'} | Device: Metal | {int(width)}x{int(height)} | "
        f"Steps: {int(steps)}{cfg_info}{cache_info}{''.join(timing_parts)}{save_info}"
    )
    return result["image"], info


def generate_image(
    prompt,
    style,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
    input_images,
    lora_file,
    lora_strength,
    anima_preset,
    auto_save,
    output_dir,
):
    # --- Apply style prefixes to prompt -----------------------------------
    for s in style:
        if "{prompt}" in s:
            prompt = s.format(prompt=prompt)
        else:
            prompt = s + prompt
    print(f"Using prompt {prompt}")

    # --- Ensure correct pipeline is loaded --------------------------------
    if "Z-Image" in model_choice and lora_file is not None and lora_file != "":
        model_choice = "Z-Image Turbo (Full - LoRA support)"

    pipe = load_pipeline(model_choice, device)

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    # --- Anima AIO Metal path ---------------------------------------------
    if current_model == ANIMA_MODEL_TYPE:
        return _generate_anima_result(
            prompt, anima_preset, steps, guidance, height, width, auto_save, output_dir
        )

    # --- LoRA -------------------------------------------------------------
    if current_model == "zimage-full" and lora_file:
        load_lora(lora_file, lora_strength, device)

    generator = _build_generator(device, int(seed))

    print_memory("Before generation")

    # --- Unified inference block ------------------------------------------
    mode = "txt2img"

    with torch.inference_mode():
        if current_model in (
            "flux2-klein-int8",
            "flux2-klein-sdnq",
            "flux2-klein-9b-sdnq",
        ):
            has_inputs = input_images is not None and len(input_images) > 0
            image, mode = (
                _generate_flux_img2img(
                    pipe,
                    prompt,
                    input_images,
                    height,
                    width,
                    steps,
                    guidance,
                    generator,
                )
                if has_inputs
                else _generate_txt2img(
                    pipe, prompt, height, width, steps, guidance, generator
                )
            )
        else:
            image, mode = _generate_txt2img(
                pipe, prompt, height, width, steps, guidance, generator
            )

    print_memory("After generation")

    # --- Memory cleanup ---------------------------------------------------
    _cleanup_memory()
    print_memory("After cache clear")

    # --- Format info + optional save --------------------------------------
    model_short = MODEL_SHORT_NAMES.get(current_model, current_model)
    _format_generation_info._lora_strength = lora_strength if lora_file else None
    info = _format_generation_info(seed, mode, guidance, lora_file, model_short)

    if auto_save:
        save_result = save_image(image, output_dir, prompt)
        info += f" | {save_result}"

    return image, info


def load_lora_and_update_strength(lora_file, lora_strength):
    """Load LoRA and persist the strength for info formatting."""
    global current_lora_path, pipe, _format_generation_info
    result = load_lora(lora_file, lora_strength, current_device)
    if lora_file is not None and lora_file != "" and "Error" not in result:
        _format_generation_info._lora_strength = lora_strength
    return result


def clear_lora():
    """Clear the current LoRA."""
    global current_lora_path, pipe
    if current_lora_path is not None and pipe is not None:
        pipe.unload_lora_weights()
        current_lora_path = None
    return None, "LoRA cleared"


# =============================================================================
# Output/Save Functions
# =============================================================================


def get_output_dir(custom_dir=None):
    """Get output directory, creating if needed."""
    output_dir = (
        custom_dir.strip() if custom_dir and custom_dir.strip() else DEFAULT_OUTPUT_DIR
    )
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_image(image, output_dir=None, prompt=""):
    """Save image to output directory."""
    if image is None:
        return "No image to save"

    output_dir = get_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = ""
    if prompt:
        prompt_slug = "_" + "".join(
            c if c.isalnum() else "_" for c in prompt[:30]
        ).strip("_")

    filename = f"{timestamp}{prompt_slug}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath, "PNG")
    return f"Saved: {filepath}"


# =============================================================================
# Storage Management Functions
# =============================================================================

# Models this app uses (HuggingFace repo IDs)
KNOWN_MODELS = {
    "aydin99/FLUX.2-klein-4B-int8": "FLUX.2-klein-4B (Int8)",
    "black-forest-labs/FLUX.2-klein-4B": "FLUX.2-klein-4B (Base)",
    "black-forest-labs/FLUX.2-klein-9B": "FLUX.2-klein-9B (Base)",
    "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic": "FLUX.2-klein-4B (4bit SDNQ)",
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32": "FLUX.2-klein-9B (4bit SDNQ)",
    "Tongyi-MAI/Z-Image-Turbo": "Z-Image Turbo (Full)",
    "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32": "Z-Image Turbo (Quantized)",
    "filipstrand/Z-Image-Turbo-mflux-4bit": "Z-Image Turbo (mflux 4bit)",
}


def get_hf_cache_dir():
    """Get HuggingFace cache directory."""
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_dir_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def scan_downloaded_models():
    """Scan HuggingFace cache for downloaded models used by this app."""
    cache_dir = get_hf_cache_dir()
    models = []
    total_size = 0

    if os.path.exists(cache_dir):
        for repo_id, display_name in KNOWN_MODELS.items():
            # Convert repo_id to cache folder name (owner--model)
            cache_name = f"models--{repo_id.replace('/', '--')}"
            model_path = os.path.join(cache_dir, cache_name)

            if os.path.exists(model_path):
                size = get_dir_size(model_path)
                total_size += size
                models.append(
                    {
                        "repo_id": repo_id,
                        "display_name": display_name,
                        "cache_name": cache_name,
                        "path": model_path,
                        "size": size,
                        "size_str": format_size(size),
                    }
                )

    anima_entry = get_anima_storage_entry()
    if anima_entry is not None:
        total_size += anima_entry["size"]
        models.append(anima_entry)

    models.sort(key=lambda x: x["size"], reverse=True)

    return models, format_size(total_size)


def get_storage_display():
    """Get formatted storage display for Gradio."""
    models, total = scan_downloaded_models()

    if not models:
        return "No models downloaded yet. Models will download on first use."

    lines = [f"**Total Storage Used: {total}**\n"]
    lines.append("| Model | Size |")
    lines.append("|-------|------|")

    for m in models:
        lines.append(f"| {m['display_name']} | {m['size_str']} |")

    return "\n".join(lines)


def get_model_choices_for_deletion():
    """Get list of model choices for deletion dropdown."""
    models, _ = scan_downloaded_models()
    choices = []
    for m in models:
        choices.append(f"{m['display_name']} ({m['size_str']})")
    return choices


def delete_model(model_selection):
    """Delete a specific model from cache."""
    global pipe, current_device, current_model

    if not model_selection:
        return (
            get_storage_display(),
            get_model_choices_for_deletion(),
            "No model selected",
        )

    models, _ = scan_downloaded_models()

    target = None
    for m in models:
        if model_selection.startswith(m["display_name"]):
            target = m
            break

    if not target:
        return (
            get_storage_display(),
            get_model_choices_for_deletion(),
            f"Model not found: {model_selection}",
        )

    # Unload pipeline if it's using this model
    model_repo = target.get("repo_id", "").lower()
    if target.get("external") == "anima" and current_model == ANIMA_MODEL_TYPE:
        current_model = None
        current_device = None

    if pipe is not None:
        needs_unload = False
        if target.get("external") == "anima" and current_model == ANIMA_MODEL_TYPE:
            needs_unload = True
        elif (
            "klein-4b" in model_repo and current_model and "4b" in current_model.lower()
        ):
            needs_unload = True
        elif (
            "klein-9b" in model_repo and current_model and "9b" in current_model.lower()
        ):
            needs_unload = True
        elif (
            "z-image" in model_repo.lower()
            and current_model
            and "zimage" in current_model.lower()
        ):
            needs_unload = True

        if needs_unload:
            del pipe
            pipe = None
            current_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    try:
        if target.get("external") == "anima":
            delete_anima_model()
        else:
            shutil.rmtree(target["path"])
        msg = f"Deleted: {target['display_name']} ({target['size_str']} freed)"
        print(msg)
    except Exception as e:
        msg = f"Error deleting {target['display_name']}: {str(e)}"
        print(msg)

    return get_storage_display(), get_model_choices_for_deletion(), msg


def delete_all_models():
    """Delete all downloaded models."""
    global pipe, current_device, current_model, current_lora_path

    models, total = scan_downloaded_models()

    if not models:
        return (
            get_storage_display(),
            get_model_choices_for_deletion(),
            "No models to delete",
        )

    if pipe is not None:
        del pipe
        pipe = None
        current_model = None
        current_device = None
        current_lora_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    deleted = []
    errors = []

    for m in models:
        try:
            if m.get("external") == "anima":
                delete_anima_model()
            else:
                shutil.rmtree(m["path"])
            deleted.append(m["display_name"])
        except Exception as e:
            errors.append(f"{m['display_name']}: {str(e)}")

    if errors:
        msg = f"Deleted {len(deleted)} models. Errors: {'; '.join(errors)}"
    else:
        msg = f"Deleted {len(deleted)} models. {total} freed."

    print(msg)
    return get_storage_display(), get_model_choices_for_deletion(), msg


def calculate_dimensions_from_ratio(
    width: int, height: int, target_resolution: str
) -> tuple:
    """Calculate output dimensions maintaining aspect ratio for target resolution."""
    if "1536" in target_resolution:
        target_size = 1536
    elif "1280" in target_resolution:
        target_size = 1280
    elif "2048" in target_resolution or "2K" in target_resolution:
        target_size = 2048
    elif "512" in target_resolution:
        target_size = 512
    else:
        target_size = 1024

    aspect_ratio = width / height

    if aspect_ratio >= 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64

    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))

    return new_width, new_height


def on_image_upload(images, current_preset):
    if images is None or len(images) == 0:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, value="~1024px"),
        )

    try:
        first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
        img_width, img_height = first_image.size
    except Exception:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, value="~1024px"),
        )

    preset = (
        current_preset
        if current_preset in ["~512px", "~1024px", "~1280px", "~1536px (32GB+)"]
        else "~1024px"
    )
    new_width, new_height = calculate_dimensions_from_ratio(
        img_width, img_height, preset
    )

    return (
        gr.update(visible=False, value=new_width),
        gr.update(visible=False, value=new_height),
        gr.update(visible=True, value=preset),
    )


def on_resolution_preset_change(preset, images):
    if images is None or len(images) == 0:
        return gr.update(), gr.update()

    first_image = images[0][0] if isinstance(images[0], tuple) else images[0]
    img_width, img_height = first_image.size
    new_width, new_height = calculate_dimensions_from_ratio(
        img_width, img_height, preset
    )

    return gr.update(value=new_width), gr.update(value=new_height)


def update_ui_for_model(model_choice):
    """Update UI visibility and defaults based on model selection."""
    is_flux = "FLUX" in model_choice
    is_zimage_full = "Full" in model_choice
    is_anima = is_anima_model_choice(model_choice)

    if is_anima:
        guidance_default = ANIMA_DEFAULTS["guidance"]
        height_default = ANIMA_DEFAULTS["height"]
        width_default = ANIMA_DEFAULTS["width"]
        steps_default = ANIMA_DEFAULTS["steps"]
    else:
        guidance_default = 3.5 if is_flux else 0.0
        height_default = 512
        width_default = 512
        steps_default = 4

    return (
        gr.update(visible=is_flux),  # img2img_label
        gr.update(visible=is_flux),  # input_image
        gr.update(visible=is_flux),  # resolution_preset
        gr.update(visible=is_zimage_full),  # lora_label
        gr.update(visible=is_zimage_full),  # lora_file
        gr.update(visible=is_zimage_full),  # lora_strength
        gr.update(visible=is_zimage_full),  # clear_lora_btn
        gr.update(visible=is_anima, value="Balanced"),  # anima_preset
        gr.update(value=guidance_default),  # guidance_scale
        # gr.update(value=height_default),  # height
        # gr.update(value=width_default),  # width
        # gr.update(value=steps_default),  # steps
    )


def update_anima_preset(preset_name):
    """Apply Anima preset defaults to visible generation controls."""
    preset = get_anima_preset(preset_name)
    return (
        gr.update(value=preset["steps"]),
        gr.update(value=ANIMA_DEFAULTS["guidance"]),
    )


def default_browser_state():
    return {
        "model": "",
        "style": [],
        "prompt": "",
        "width": 512,
        "height": 512,
        "auto_save": False,
        "steps": 4,
        "seed": -1,
    }


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
with gr.Blocks(title="Ultra Fast Image Gen") as demo:
    gr.Markdown("""
    # Ultra Fast Image Gen
    
    AI image generation and editing on Apple Silicon and CUDA.
    
    **Models:**
    - **FLUX.2-klein-4B (Int8):** 8GB, supports image-to-image editing (default)
    - **Z-Image Turbo (Quantized):** 3.5GB, fastest, no LoRA
    - **Anima Turbo AIO Q4 (Metal):** local patched sd.cpp runner, defaults to 512x768 / 8 steps
    - **Z-Image Turbo (Full):** 24GB, slower, LoRA support
    
    **Resolutions:** Up to 2048px for txt2img. Image-to-image: 1K (16GB) or 1.5K (32GB+).
    """)

    browser_state = gr.BrowserState(
        default_browser_state(),
        storage_key="ultra-fast-image-gen",
        secret="user-data",
    )

    styles = []
    if os.path.exists("styles_integrated.csv"):
        with open("styles_integrated.csv", "r") as f:
            style_reader = csv.reader(f)
            for row in style_reader:
                styles.append({"name": row[0], "prompt": row[1]})

    with gr.Row():
        with gr.Column(scale=1):
            # Model selection
            model_choice = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Model",
                info="FLUX.2-klein supports image editing",
            )

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )

            style = gr.Dropdown(
                choices=[(x["name"], x["prompt"]) for x in styles],
                label="Styles",
                multiselect=True,
                visible=len(styles) > 0,
            )

            # Image input (FLUX only) - visible by default since FLUX is default
            img2img_label = gr.Markdown(
                "### Image Input (FLUX.2-klein only - up to 6 images)", visible=True
            )
            input_images = gr.Gallery(
                label="Input Images (optional - for image-to-image)",
                type="pil",
                visible=True,
                columns=3,
                height="auto",
                interactive=True,
            )

            resolution_preset = gr.Radio(
                choices=["~512px", "~1024px", "~1280px", "~1536px (32GB+)"],
                value="~1024px",
                label="Output Resolution (longest side)",
                info="Maintains your image's aspect ratio",
                visible=False,
            )

            with gr.Row():
                height = gr.Slider(256, 2048, value=512, step=64, label="Height")
                width = gr.Slider(256, 2048, value=512, step=64, label="Width")

            with gr.Row():
                steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            anima_preset = gr.Radio(
                choices=list(ANIMA_PRESETS.keys()),
                value="Balanced",
                label="Anima Mode",
                info="Fast: 3 steps + Spectrum, Balanced: 8 + Spectrum, Quality: 16 without cache",
                visible=False,
            )

            with gr.Row():
                guidance_scale = gr.Slider(
                    0.0,
                    10.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale (CFG)",
                    info="FLUX: 3.5 recommended, Z-Image: 0",
                )

            with gr.Row():
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    info="MPS=Mac, CUDA=NVIDIA, CPU=slow",
                )

            # LoRA section (Z-Image Full only) - no Group wrapper for visibility to work
            lora_label = gr.Markdown(
                "### LoRA Settings (Z-Image Full only)", visible=False
            )
            with gr.Row():
                lora_file = gr.File(
                    label="LoRA File",
                    file_types=[".safetensors"],
                    file_count="single",
                    type="filepath",
                    visible=False,
                )
                clear_lora_btn = gr.Button(
                    "Clear LoRA", scale=0, min_width=100, visible=False
                )

            lora_strength = gr.Slider(
                0.0,
                2.0,
                value=1.0,
                step=0.05,
                label="LoRA Strength",
                info="1.0 = full effect, 0.5 = half effect",
                visible=False,
            )

            generate_btn = gr.Button("Generate", variant="primary")
            seed_info = gr.Textbox(label="Generation Info", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    with gr.Accordion("Save Settings", open=False):
        auto_save = gr.Checkbox(label="Auto-save generated images", value=False)
        output_dir = gr.Textbox(label="Output Directory", value=DEFAULT_OUTPUT_DIR)
        with gr.Row():
            save_btn = gr.Button("Save Current Image")
            open_folder_btn = gr.Button("Open Output Folder")
        save_status = gr.Textbox(label="Save Status", interactive=False)

    with gr.Accordion("Storage Management", open=False):
        storage_display = gr.Markdown(value=get_storage_display())

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=get_model_choices_for_deletion(),
                label="Select Model to Delete",
                scale=3,
            )
            delete_btn = gr.Button("Delete Selected", variant="secondary", scale=1)

        with gr.Row():
            refresh_btn = gr.Button("Refresh", scale=1)
            delete_all_btn = gr.Button("Delete ALL Models", variant="stop", scale=1)

        storage_status = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=[
            ["A majestic mountain landscape at sunset, dramatic lighting, cinematic"],
            [
                "Portrait of a young woman, soft studio lighting, professional photography"
            ],
            ["Cyberpunk city street at night, neon lights, rain reflections"],
            ["A cute cat wearing a tiny hat, studio photo, soft lighting"],
            ["Abstract art, vibrant colors, fluid shapes, modern design"],
        ],
        inputs=[prompt],
    )

    # Event handlers
    SAVE_INPUTS = [model_choice, prompt, style, width, height, steps, auto_save, seed]
    SAVE_OUTPUTS = [browser_state]

    def save_settings(*args):
        return dict(zip(
            ["model", "prompt", "style", "width", "height", "steps", "auto_save", "seed"],
            args,
        ))

    for component in SAVE_INPUTS:
        component.change(fn=save_settings, inputs=SAVE_INPUTS, outputs=SAVE_OUTPUTS)

    model_choice.change(
        fn=update_ui_for_model,
        inputs=[model_choice],
        outputs=[
            img2img_label,
            input_images,
            resolution_preset,
            lora_label,
            lora_file,
            lora_strength,
            clear_lora_btn,
            anima_preset,
            guidance_scale,
            # height,
            # width,
            # steps,
        ],
    )

    anima_preset.change(
        fn=update_anima_preset,
        inputs=[anima_preset],
        outputs=[steps, guidance_scale],
    )

    input_images.change(
        fn=on_image_upload,
        inputs=[input_images, resolution_preset],
        outputs=[width, height, resolution_preset],
    )

    resolution_preset.change(
        fn=on_resolution_preset_change,
        inputs=[resolution_preset, input_images],
        outputs=[width, height],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            style,
            height,
            width,
            steps,
            seed,
            guidance_scale,
            device,
            model_choice,
            input_images,
            lora_file,
            lora_strength,
            anima_preset,
            auto_save,
            output_dir,
        ],
        outputs=[output_image, seed_info],
    )

    def manual_save(image, out_dir, prompt_text):
        if image is None:
            return "No image to save"
        result = save_image(image, out_dir, prompt_text)
        return result

    save_btn.click(
        fn=manual_save,
        inputs=[output_image, output_dir, prompt],
        outputs=[save_status],
    )

    def open_output_folder(out_dir):
        import subprocess

        folder = get_output_dir(out_dir)
        subprocess.run(["open", folder])
        return f"Opened: {folder}"

    open_folder_btn.click(
        fn=open_output_folder,
        inputs=[output_dir],
        outputs=[save_status],
    )

    clear_lora_btn.click(
        fn=clear_lora,
        outputs=[lora_file, seed_info],
    )

    lora_strength.change(
        fn=update_lora_strength,
        inputs=[lora_strength],
        outputs=[seed_info],
    )

    def refresh_storage():
        return get_storage_display(), get_model_choices_for_deletion(), ""

    refresh_btn.click(
        fn=refresh_storage,
        outputs=[storage_display, model_dropdown, storage_status],
    )

    delete_btn.click(
        fn=delete_model,
        inputs=[model_dropdown],
        outputs=[storage_display, model_dropdown, storage_status],
    )

    delete_all_btn.click(
        fn=delete_all_models,
        outputs=[storage_display, model_dropdown, storage_status],
    )

    def load_settings(saved_settings):
        if saved_settings is None:
            saved_settings = default_browser_state()
        return (
            saved_settings["model"],
            saved_settings["prompt"],
            saved_settings["style"],
            saved_settings["width"],
            saved_settings["height"],
            saved_settings["steps"],
            saved_settings["auto_save"],
            saved_settings["seed"],
        )

    demo.load(
        fn=load_settings,
        inputs=[browser_state],
        outputs=[model_choice, prompt, style, width, height, steps, auto_save, seed],
    )

if __name__ == "__main__":
    demo.launch()
