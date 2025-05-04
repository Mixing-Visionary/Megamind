import requests
import torch
from app.core.config import settings
import gc
from app.models.enums import ProcessingStyle
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, \
    AutoencoderKL, AutoPipelineForText2Image
from transformers import BlipProcessor, BlipForConditionalGeneration


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    def load_base_model(self):
        """Loads the base model if not already loaded"""
        if not self.base_model:
            self.base_model = AutoPipelineForImage2Image.from_pretrained(
                settings.BASE_MODEL_PATH,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=settings.SAFETY_CHECKER
            ).to(self.device)

            print(self.device)
        if not self.prompt_processor:
            self.prompt_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
        if not self.prompt_model:
            self.prompt_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                torch_dtype=self.dtype
            ).to("cuda")
        return self.base_model, self.prompt_processor, self.prompt_model

