import base64
import time
import asyncio
from io import BytesIO

import cv2
from PIL import Image, ImageOps
import numpy as np
from fastapi import HTTPException
from diffusers.utils import load_image

from app.core.config import settings
from app.models.enums import ProcessingStyle
from app.models.schemas import ImageProcessingRequest, ImageProcessingResponse
from app.services.model_loader import ModelLoader


class ImageProcessor:
    def __init__(self):
        self.model_loader = ModelLoader()
        self._processing_lock = asyncio.Lock()  # For sequential processing

    async def process_image(self, request: ImageProcessingRequest) -> ImageProcessingResponse:
        """Process an image with error handling and validation"""
        start_time = time.time()

        # Validate image data
        init_image = self._decode_image(request.image)

        # Save init size
        print(f"Init image stats - Size: {init_image.size}, Mode: {init_image.mode}")
        init_size = init_image.size
        self._check_black_image(init_image)

        # Check image dimensions
        init_image = self._validate_dimensionality(init_image)
        init_image = load_image(init_image)

        # Sequential processing
        async with self._processing_lock:
            try:
                # Load model
                pipe, prompt_processor, prompt_model = self.model_loader.load_base_model()
                style = ProcessingStyle(request.style)
                pipe = self.model_loader.set_lora_model(request.style)

                # Generate mask
                mask = self._get_mask(init_image, self.model_loader)

                # Prepare image
                init_image = self._prepare_image(init_image, style)

                # Generate caption
                caption = self._generate_prompt(init_image, prompt_processor, prompt_model)

                prompt = f"{caption}"
                negative = style.get_negative_prompt()
                print(f"Prompt: {prompt}")
                print(init_image.size)


                # Process image with proper error handling
                print(f'init {init_image.size}')
                print(f'depth {mask.size}')
                if request.strength > np.exp(-3):
                    result = await self._run_inference(pipe, init_image, request, prompt, negative, mask)
                else:
                    result = init_image
                # Encode result
                result = result.resize(init_size, Image.Resampling.LANCZOS)
                processed_image = self._encode_image(result)

                print(f"Result image stats - Size: {result.size}, Mode: {result.mode}")

                self._check_black_image(init_image)

                return ImageProcessingResponse(
                    processed_image=processed_image,
                    processing_time=time.time() - start_time,
                    style=request.style.value
                )
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Detailed error: {error_trace}")
                raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

    def _validate_dimensionality(self, image):
        if max(image.size) > settings.MAX_IMAGE_SIZE:
            image = self._resize_image(image, settings.MAX_IMAGE_SIZE)
        return image

    def _decode_image(self, data) -> Image:
        try:
            image_data = base64.b64decode(data)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"Invalid image data: {str(e)}")
        return image
    def _encode_image(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_mask(self, image, model_loader):
        if settings.MASK_STRATEGY == "depth":
            return self._get_depth_map(image, model_loader.get_depth_detector())
        else:
            return self._get_canny(image)

    def _prepare_image(self, image: Image, style: ProcessingStyle) -> Image:
        if style is ProcessingStyle.NOIR:
            return ImageOps.grayscale(image)
        return image

    def _generate_prompt(self, image: Image, prompt_processor, prompt_model) -> str:
        inputs = prompt_processor(image, return_tensors="pt").to("cuda")
        out = prompt_model.generate(**inputs)
        return prompt_processor.decode(out[0], skip_special_tokens=True)

    def _get_canny(self, image: Image) -> Image:
        init_size = image.size
        image = np.array(image)
        image = cv2.Canny(image, 60, 100)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image).resize(init_size, Image.Resampling.LANCZOS)

    def _get_depth_map(self, image: Image, depth_detector) -> Image:
        init_size = image.size
        return depth_detector(image).resize(init_size, Image.Resampling.LANCZOS)

    def _check_black_image(self, image: Image) -> None:
        result_array = np.array(image)
        if np.mean(result_array) < 0.01:
            print("Init image is black or nearly black!")

    async def _run_inference(self, pipe, image, request, prompt, negative, controlnet_image):
        """Run inference in a non-blocking way"""
        # Create a loop in a thread to run the model inference
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=f"{request.style} style, {prompt}",
                negative_prompt=negative,
                image=image,
                strength=request.strength,
                guidance_scale=settings.DEFAULT_GUIDANCE_SCALE,
                num_inference_steps=settings.DEFAULT_STEPS,
                aesthetic_score=6,
                negative_aesthetic_score=6,

                controlnet_conditioning_scale=settings.DEFAULT_CONTROLNET_SCALE,
                control_image=controlnet_image,
            ).images[0]
        )

    def _resize_image(self, image, max_size) -> Image:
        """Resize image maintaining aspect ratio"""
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
