import base64
import time
import asyncio
from io import BytesIO

import PIL
from PIL import Image
import numpy as np
from fastapi import HTTPException
from diffusers.utils import load_image

from app.core.config import settings
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
        try:
            image_data = base64.b64decode(request.image)
            init_image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"Invalid image data: {str(e)}")
        print(f"Init image stats - Size: {init_image.size}, Mode: {init_image.mode}")
        init_size = init_image.size
        # Check for black image
        result_array = np.array(init_image)
        if np.mean(result_array) < 0.01:  # If image is mostly black
            print("Init image is black or nearly black!")

        # Check image dimensions
        if max(init_image.size) > settings.MAX_IMAGE_SIZE:
            # Resize image to maintain aspect ratio
            init_image = self._resize_image(init_image, settings.MAX_IMAGE_SIZE)
        init_image = load_image(init_image)
        # Sequential processing for GPU memory management
        async with self._processing_lock:
            try:
                # Load model (singleton pattern ensures efficient loading)
                pipe, prompt_processor, prompt_model, canny_detector, upscaler = self.model_loader.load_base_model()
                pipe = self.model_loader.set_lora_model(request.style.value)

                # Generating prompt
                inputs = prompt_processor(init_image, return_tensors="pt")
                out = prompt_model.generate(**inputs, max_new_tokens=100)
                prompt = prompt_processor.decode(out[0], skip_special_tokens=True)
                print(f"Prompt: {prompt}")
                canny_image = canny_detector(
                    init_image,
                    low_threshold=100,
                    high_threshold=300,
                    image_resolution=min(init_image.size)
                )
                canny_image = canny_image.resize(init_image.size, Image.LANCZOS)
                # canny_image = PIL.ImageOps.invert(canny_image)
                canny_image.save('canny.jpeg')
                print(init_image.size)
                print(canny_image.size)
                # Process image with proper error handling
                result = await self._run_inference(pipe, init_image, request, prompt, canny_image)
                result = upscaler(prompt=prompt, image=result, num_inference_steps=25).images[0]

                result = self._resize_image(result, max(init_size))
                # Encode result
                buffered = BytesIO()
                result.save(buffered, format="JPEG", quality=95)
                processed_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

                print(f"Result image stats - Size: {result.size}, Mode: {result.mode}")
                # Check for black image
                result_array = np.array(result)
                if np.mean(result_array) < 0.01:  # If image is mostly black
                    print("Output image is black or nearly black!")

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

    async def _run_inference(self, pipe, image, request, prompt, canny_image):
        """Run inference in a non-blocking way"""
        # Create a loop in a thread to run the model inference
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=f"{request.style} style, {prompt}",
                image=image,
                control_image=canny_image,
                controlnet_conditioning_scale=settings.DEFAULT_CONTROLNET_SCALE,
                strength=request.strength,
                guidance_scale=settings.DEFAULT_GUIDANCE_SCALE,
                num_inference_steps=settings.DEFAULT_STEPS
            ).images[0]
        )

    def _resize_image(self, image, max_size):
        """Resize image maintaining aspect ratio"""
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        return image.resize((new_width, new_height), Image.LANCZOS)
