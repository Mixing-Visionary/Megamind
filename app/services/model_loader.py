import requests
import torch
from controlnet_aux import MidasDetector

from app.core.config import settings
import gc
from app.models.enums import ProcessingStyle
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, \
    AutoencoderKL, AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, ControlNetModel, \
    StableDiffusionXLControlNetImg2ImgPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.base_model = None
        self.prompt_processor = None
        self.prompt_model = None
        self.loras_loaded = False
        self.depth_detector = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if settings.USE_HALF_PRECISION else torch.float32
        self.variant = "fp16" if settings.USE_HALF_PRECISION else "fp32"

    def load_base_model(self):
        """Loads the base model if not already loaded"""
        if not self.base_model:
            if settings.MASK_STRATEGY == "depth":
                controlnet_model = settings.CONTROLNET_DEPTH_MODEL
            else:
                controlnet_model = settings.CONTROLNET_CANNY_MODEL
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model,
                torch_dtype=self.dtype,
                variant=self.variant,
                use_safetensors=True
            ).to(self.device)

            vae = AutoencoderKL.from_pretrained(settings.VAE, torch_dtype=self.dtype).to(self.device)

            self.base_model = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                settings.BASE_MODEL,
                torch_dtype=self.dtype,
                controlnet=controlnet,
                vae=vae,
                variant=self.variant,
                use_safetensors=True,
                safety_checker=settings.SAFETY_CHECKER
            ).to(self.device)

            print(self.device)

        if not self.prompt_processor:
            self.prompt_processor = BlipProcessor.from_pretrained(
                settings.PROMPT_MODEL
            )
        if not self.prompt_model:
            self.prompt_model = BlipForConditionalGeneration.from_pretrained(
                settings.PROMPT_MODEL,
                torch_dtype=self.dtype
            ).to(self.device)

        self.depth_detector = self.get_depth_detector()

        if settings.ENABLE_MODEL_CPU_OFFLOAD:
            self.base_model.enable_model_cpu_offload()

        if not self.loras_loaded:
            self._load_loras()
            self.loras_loaded = True

        return self.base_model, self.prompt_processor, self.prompt_model

    def get_depth_detector(self):
        if not self.depth_detector:
            self.depth_detector = MidasDetector.from_pretrained(settings.DEPTH_DETECTOR)
        return self.depth_detector

    def _load_loras(self):
        """Loads a LoRA models"""
        for style in ProcessingStyle:
            self.base_model.load_lora_weights(style.get_path(), adapter_name=style.value)
            self.base_model.set_adapters([style.value], adapter_weights=0)

    def set_lora_model(self, style: str):
        """Set a LoRA model"""
        self.current_lora_style = style
        self._reset_loras()
        self.base_model.set_adapters([style], adapter_weights=1.0)

        return self.base_model

    def _reset_loras(self):
        for style in ProcessingStyle:
            self.base_model.set_adapters([style.value], adapter_weights=0)