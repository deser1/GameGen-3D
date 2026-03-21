import cv2
import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class TextureUpscaler:
    def __init__(self, scale: int = 2):
        """
        Inicjalizuje model Real-ESRGAN do powiększania tekstur 
        (np. z 512x512 do 1024x1024 lub 2048x2048) z zachowaniem ostrości detali.
        """
        print(f"[Init] Ładowanie modelu Real-ESRGAN (Upscaler x{scale})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale = scale
        
        try:
            # Używamy modelu do ogólnych obrazów (plus2), który dobrze radzi sobie z teksturami
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            # URL do wag modelu RealESRGAN_x2plus
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            
            self.upsampler = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=0, # tile size, 0 = cały obraz na raz
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
        except Exception as e:
            print(f"[Błąd] Nie udało się załadować Real-ESRGAN. Błąd: {e}")
            self.upsampler = None

    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Powiększa obraz (teksturę) x2 lub x4.
        """
        if self.upsampler is None:
            # Fallback - zwykłe skalowanie dwuliniowe
            new_size = (image.width * self.scale, image.height * self.scale)
            return image.resize(new_size, Image.LANCZOS)
            
        print(f"  [Upscaler] Powiększanie tekstury {image.width}x{image.height} -> {image.width * self.scale}x{image.height * self.scale}...")
        
        # Konwersja PIL -> OpenCV (BGR)
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        try:
            # Upscaling
            output, _ = self.upsampler.enhance(cv_img, outscale=self.scale)
            # Konwersja z powrotem na PIL (RGB)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output_rgb)
        except Exception as e:
            print(f"  [Upscaler Błąd] {e}. Wracam do skalowania podstawowego.")
            new_size = (image.width * self.scale, image.height * self.scale)
            return image.resize(new_size, Image.LANCZOS)
