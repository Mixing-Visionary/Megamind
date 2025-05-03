import torch
from app.core.config import settings
import gc
from app.models.enums import ProcessingStyle
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, \
    AutoencoderKL, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLPipeline, \
    StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from controlnet_aux import CannyDetector
from pathlib import Path


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.base_model = None
        self.current_lora_style = None
        self.prompt_processor = None
        self.prompt_model = None
        self.canny_detector = None
        self.controlnet_model = None
        self.upscaler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

    def load_base_model(self):
        """Loads the base model if not already loaded"""
        if not self.base_model:

            controlnet_model = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=self.dtype
            ).to(self.device)

            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=self.dtype
            ).to(self.device)

            self.base_model = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                settings.BASE_MODEL_PATH,
                torch_dtype=torch.float32,
                controlnet=controlnet_model,
                vae=vae,
                use_safetensors=True,
                safety_checker=settings.SAFETY_CHECKER
            )

            self.base_model = self.base_model.to(torch.float16).to(self.device)

            print(self.device)
            if settings.ENABLE_MODEL_CPU_OFFLOAD and self.device == "cuda":
                self.base_model.enable_model_cpu_offload()
                self.base_model.enable_xformers_memory_efficient_attention()
            if settings.ENABLE_ATTENTION_SLICING:
                self.base_model.enable_attention_slicing()
            if settings.DISABLE_VAE_TILING:
                self.base_model.disable_vae_tiling()
            if settings.DISABLE_VAE_SLICING:
                self.base_model.disable_vae_slicing()
            self._load_loras()
        if not self.upscaler:
            self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
                "stabilityai/sd-x2-latent-upscaler",
                torch_dtype=self.dtype
            ).to(self.device)
        if not self.prompt_processor:
            self.prompt_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        if not self.prompt_model:
            self.prompt_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        if not self.canny_detector:
            self.canny_detector = CannyDetector()
        return self.base_model, self.prompt_processor, self.prompt_model, self.canny_detector, self.upscaler

    def _load_loras(self):
        """Loads a LoRA models"""
        for style in ProcessingStyle:
            self.base_model.load_lora_weights(style.get_path(), adapter_name=style.value)
            self.base_model.set_adapters([style.value], adapter_weights=0)

    def _reset_loras(self):
        for style in ProcessingStyle:
            self.base_model.set_adapters([style.value], adapter_weights=0)

    def set_lora_model(self, style: str):
        """Set a LoRA model"""
        self.current_lora_style = style
        self._reset_loras()
        self.base_model.set_adapters([style], adapter_weights=1.0)

        return self.base_model
