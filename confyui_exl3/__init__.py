from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from exllamav3 import Config, Model, Tokenizer, Generator
from exllamav3.cache import Cache, CacheLayer_fp16, CacheLayer_quant
from exllamav3.generator.sampler.presets import ComboSampler


TEXT_PRESETS: dict[str, str] = {
    "Prompt expander (neutral)": "You are a prompt expander. Expand the user prompt into a detailed, clear, and structured prompt for a text-to-image model. Keep style consistent and avoid adding new entities.",
    "Prompt expander (cinematic)": "You are a prompt expander specializing in cinematic visuals. Expand the user prompt with camera, lighting, mood, and environment details without adding new subjects.",
    "Prompt expander (product photo)": "You are a prompt expander for product photography. Expand the prompt with studio lighting, lens, background, and material details. Keep it commercial and clean.",
    "Prompt expander (editorial fashion)": "You are a prompt expander for editorial fashion. Expand the prompt with styling, composition, lighting, and magazine-grade art direction.",
    "Prompt expander (architectural)": "You are a prompt expander for architectural visualization. Expand the prompt with materials, lighting, time of day, and spatial detail.",
    "Prompt expander (fantasy)": "You are a prompt expander for fantasy worlds. Expand the prompt with atmosphere, scale, biomes, and narrative hints without adding extra characters.",
    "Prompt expander (sci-fi)": "You are a prompt expander for science fiction. Expand the prompt with technology, materials, lighting, and futuristic ambience.",
    "Prompt expander (character sheet)": "You are a prompt expander for character sheets. Expand with clothing, materials, colors, expression, and turnaround angles.",
    "Prompt expander (game asset)": "You are a prompt expander for game art assets. Expand with material definition, topology hints, and neutral presentation.",
    "Prompt expander (photojournalism)": "You are a prompt expander for photojournalistic images. Expand with realistic lighting, candid framing, and documentary tone.",
    "Prompt expander (minimalist)": "You are a prompt expander for minimalist design. Expand with clean composition, negative space, and restrained palette.",
    "Prompt expander (macro)": "You are a prompt expander for macro photography. Expand with lens specs, depth of field, micro detail, and lighting.",
    "Prompt expander (food)": "You are a prompt expander for food photography. Expand with plating, lighting, background, and texture cues.",
    "Prompt expander (interior)": "You are a prompt expander for interior design. Expand with materials, lighting, and composition, preserving the original layout.",
    "Prompt expander (illustration)": "You are a prompt expander for illustration. Expand with line quality, palette, and style references while preserving subject.",
    "Prompt expander (concept art)": "You are a prompt expander for concept art. Expand with storytelling, environment design, and mood.",
    "Prompt expander (noir)": "You are a prompt expander for noir visuals. Expand with chiaroscuro lighting, moody atmosphere, and period detail.",
    "Prompt expander (nature)": "You are a prompt expander for nature photography. Expand with weather, time of day, and environmental details.",
    "Prompt expander (isometric)": "You are a prompt expander for isometric art. Expand with layout, scale, and clear object separation.",
    "Prompt expander (technical diagram)": "You are a prompt expander for technical diagrams. Expand with labels, clean lines, and precise composition.",
}

IMAGE_CAPTION_PRESETS: dict[str, str] = {
    "Caption (detailed)": "Describe the image in rich detail: subjects, actions, environment, style, lighting, camera angle, and mood.",
    "Caption (short)": "Provide a concise one-sentence caption describing the core subject and action.",
    "Caption (tags)": "Return a comma-separated list of visual tags, materials, colors, styles, and notable objects.",
    "Caption (product listing)": "Write a product listing description: include materials, colors, usage context, and quality cues.",
    "Caption (art critique)": "Analyze the image as an art critic: composition, color palette, lighting, and mood.",
    "Caption (scene analysis)": "Analyze the scene: spatial layout, objects, and implied narrative context.",
    "Caption (photography)": "Describe the photographic setup: lens feel, depth of field, lighting, and framing.",
    "Caption (palette)": "List the dominant colors and color temperature as a palette description.",
    "Caption (accessibility)": "Write an accessibility-focused description for screen readers.",
    "Caption (OCR)": "Extract all legible text from the image. If no text is visible, answer: 'NO TEXT'.",
    "Caption (layout)": "Describe the layout and composition structure: foreground, midground, background.",
    "Caption (material focus)": "Describe the materials, textures, and surface finishes visible.",
}

CLIP_PROMPT_PRESETS: dict[str, str] = {
    "Qwen Image Edit (balanced)": "Rewrite the prompt for Qwen Image Edit with balanced specificity and fidelity. Preserve the subject and keep instructions edit-focused.",
    "Qwen Image Edit (precise)": "Rewrite the prompt for Qwen Image Edit with precise edit instructions, including what must remain unchanged.",
    "Qwen Image Edit (creative)": "Rewrite the prompt for Qwen Image Edit with creative direction, while keeping the original subject intact.",
    "Z Image Turbo (balanced)": "Rewrite the prompt for Z Image Turbo with balanced detail, emphasizing clarity and artistic intent.",
    "Z Image Turbo (stylized)": "Rewrite the prompt for Z Image Turbo with stylized art direction, including palette, lighting, and mood.",
    "Z Image Turbo (photoreal)": "Rewrite the prompt for Z Image Turbo with photorealistic constraints and camera cues.",
    "Z Image Turbo (fast)": "Rewrite the prompt for Z Image Turbo into a concise but clear edit instruction.",
}


@dataclass
class Exl3ModelBundle:
    model: Model
    tokenizer: Tokenizer
    cache: Cache
    generator: Generator
    vision_model: Model | None = None


class Exl3ClipPrompt:
    def __init__(self, prompt: str, negative_prompt: str, metadata: dict[str, Any]):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.metadata = metadata

    def tokenize(self, text: str) -> dict[str, Any]:
        return {"text": text}

    def encode_from_tokens(self, tokens: dict[str, Any], return_pooled: bool = True) -> dict[str, Any]:
        return {
            "text": tokens.get("text", ""),
            "negative_text": self.negative_prompt,
            "metadata": self.metadata,
        }


def _resolve_preset(presets: dict[str, str], preset_name: str, custom_prompt: str) -> str:
    preset_prompt = presets.get(preset_name, "")
    if custom_prompt and preset_prompt:
        return f"{preset_prompt}\n\nCustom instructions: {custom_prompt}"
    if custom_prompt:
        return custom_prompt
    return preset_prompt


def _tensor_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 4:
            tensor = tensor[0]
        tensor = tensor.detach().cpu().float().clamp(0, 1)
        array = (tensor.numpy() * 255.0).round().astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array)
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    raise TypeError("Unsupported image type for captioning")


def _build_sampler(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> ComboSampler:
    return ComboSampler(
        rep_p=repetition_penalty,
        freq_p=frequency_penalty,
        pres_p=presence_penalty,
        temperature=temperature,
        min_p=min_p,
        top_k=top_k,
        top_p=top_p,
    )


def _generate_text(
    bundle: Exl3ModelBundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: list[str],
    embeddings: list[Any] | None = None,
) -> str:
    sampler = _build_sampler(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    completion = bundle.generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        sampler=sampler,
        stop_conditions=stop,
        completion_only=True,
        embeddings=embeddings,
    )
    return completion.strip()


class Exl3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model_dir": ("STRING", {"default": ""}),
                "cache_size": ("INT", {"default": 8192, "min": 0, "max": 262144}),
                "cache_quant": ("STRING", {"default": ""}),
                "gpu_split": ("STRING", {"default": "auto"}),
                "tensor_parallel": ("BOOLEAN", {"default": False}),
                "tp_backend": ("STRING", {"default": "native"}),
                "load_vision": ("BOOLEAN", {"default": True}),
                "max_chunk_size": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                "max_batch_size": ("INT", {"default": 8, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("EXL3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "exllamav3/exl3"

    def load(
        self,
        model_dir: str,
        cache_size: int,
        cache_quant: str,
        gpu_split: str,
        tensor_parallel: bool,
        tp_backend: str,
        load_vision: bool,
        max_chunk_size: int,
        max_batch_size: int,
    ) -> tuple[Exl3ModelBundle]:
        config = Config.from_directory(model_dir)
        model = Model.from_config(config, "text")

        cache = None
        if cache_size > 0:
            if cache_quant:
                split = [int(bits) for bits in cache_quant.split(",") if bits.strip()]
                if len(split) == 1:
                    k_bits = v_bits = split[0]
                elif len(split) == 2:
                    k_bits, v_bits = split
                else:
                    raise ValueError("cache_quant must be one value or k_bits,v_bits")
                cache = Cache(model, max_num_tokens=cache_size, layer_type=CacheLayer_quant, k_bits=k_bits, v_bits=v_bits)
            else:
                cache = Cache(model, max_num_tokens=cache_size, layer_type=CacheLayer_fp16)

        if gpu_split == "auto" or not gpu_split:
            split = None
        else:
            split = [float(alloc) for alloc in gpu_split.split(",")]

        model.load(
            use_per_device=split,
            tensor_p=tensor_parallel,
            tp_backend=tp_backend,
            max_chunk_size=max_chunk_size,
            max_output_size=max_chunk_size,
            max_output_factor=1,
            progressbar=False,
        )

        vision_model = None
        if load_vision and "vision" in config.model_classes:
            vision_model = Model.from_config(config, "vision")
            vision_model.load(
                use_per_device=split,
                tensor_p=tensor_parallel,
                tp_backend=tp_backend,
                max_chunk_size=max_chunk_size,
                max_output_size=max_chunk_size,
                max_output_factor=1,
                progressbar=False,
            )

        tokenizer = Tokenizer.from_config(config)
        if cache is None:
            cache = Cache(model, max_num_tokens=8192, layer_type=CacheLayer_fp16)

        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            max_chunk_size=max_chunk_size,
        )

        return (Exl3ModelBundle(model=model, tokenizer=tokenizer, cache=cache, generator=generator, vision_model=vision_model),)


class Exl3ImageCaptioner:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("EXL3_MODEL",),
                "image": ("IMAGE",),
                "preset": (list(IMAGE_CAPTION_PRESETS.keys()),),
                "custom_prompt": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 128, "min": 8, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 512}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.8, "max": 2.0}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "exllamav3/exl3"

    def caption(
        self,
        model: Exl3ModelBundle,
        image: Any,
        preset: str,
        custom_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> tuple[str]:
        if model.vision_model is None:
            raise ValueError("This model was loaded without a vision component.")

        pil_image = _tensor_to_pil(image)
        prompt_instruction = _resolve_preset(IMAGE_CAPTION_PRESETS, preset, custom_prompt)
        image_alias = "<image>"
        raw_embeddings = model.vision_model.get_image_embeddings(model.tokenizer, pil_image, text_alias=image_alias)
        embedding_list = raw_embeddings if isinstance(raw_embeddings, list) else [raw_embeddings]
        full_prompt = f"{prompt_instruction}\n\nImage: {image_alias}"

        caption = _generate_text(
            bundle=model,
            prompt=full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=[],
            embeddings=embedding_list,
        )

        return (caption,)


class Exl3PromptWriter:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("EXL3_MODEL",),
                "base_prompt": ("STRING", {"default": ""}),
                "preset": (list(TEXT_PRESETS.keys()),),
                "custom_prompt": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 512}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.8, "max": 2.0}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "write"
    CATEGORY = "exllamav3/exl3"

    def write(
        self,
        model: Exl3ModelBundle,
        base_prompt: str,
        preset: str,
        custom_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> tuple[str]:
        preset_prompt = _resolve_preset(TEXT_PRESETS, preset, custom_prompt)
        full_prompt = f"{preset_prompt}\n\nUser prompt: {base_prompt}\n\nExpanded prompt:".strip()

        output = _generate_text(
            bundle=model,
            prompt=full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=[],
        )

        return (output,)


class Exl3ClipPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("EXL3_MODEL",),
                "prompt": ("STRING", {"default": ""}),
                "preset": (list(CLIP_PROMPT_PRESETS.keys()),),
                "custom_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 192, "min": 16, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 512}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.8, "max": 2.0}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING", "STRING")
    RETURN_NAMES = ("clip", "clip_prompt", "clip_negative")
    FUNCTION = "build"
    CATEGORY = "exllamav3/exl3"

    def build(
        self,
        model: Exl3ModelBundle,
        prompt: str,
        preset: str,
        custom_prompt: str,
        negative_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> tuple[Exl3ClipPrompt, str, str]:
        preset_prompt = _resolve_preset(CLIP_PROMPT_PRESETS, preset, custom_prompt)
        full_prompt = f"{preset_prompt}\n\nUser prompt: {prompt}\n\nRewrite:".strip()

        rewritten = _generate_text(
            bundle=model,
            prompt=full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=[],
        )

        clip = Exl3ClipPrompt(
            prompt=rewritten,
            negative_prompt=negative_prompt,
            metadata={
                "preset": preset,
                "custom_prompt": custom_prompt,
            },
        )

        return (clip, rewritten, negative_prompt)


NODE_CLASS_MAPPINGS = {
    "EXL3 Model Loader": Exl3ModelLoader,
    "EXL3 Image Captioner": Exl3ImageCaptioner,
    "EXL3 Prompt Writer": Exl3PromptWriter,
    "EXL3 Clip Prompt Builder": Exl3ClipPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EXL3 Model Loader": "EXL3 Model Loader (exllamav3)",
    "EXL3 Image Captioner": "EXL3 Image Captioner (exllamav3)",
    "EXL3 Prompt Writer": "EXL3 Prompt Writer (exllamav3)",
    "EXL3 Clip Prompt Builder": "EXL3 Clip Prompt Builder (Qwen/Z)",
}
