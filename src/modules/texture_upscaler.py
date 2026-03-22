import cv2
import numpy as np
from PIL import Image

class TextureUpscaler:
    def __init__(self, scale: int = 2):
        """
        Inicjalizuje upscaler. Z powodu problematycznych zależności (basicsr/RealESRGAN) na Windows,
        używamy wysokiej jakości wbudowanego w OpenCV upscalera algorytmicznego (LANCZOS4)
        jako lekkiej i bezbłędnej alternatywy dla gamedevu.
        """
        print(f"[Init] Ładowanie modułu Upscalera (x{scale})...")
        self.scale = scale
        self.upsampler = None # Zostawiamy dla kompatybilności API

    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Powiększa obraz (teksturę) x2 lub x4 za pomocą wysokiej jakości algorytmu Lanczos.
        """
        print(f"  [Upscaler] Powiększanie tekstury {image.width}x{image.height} -> {image.width * self.scale}x{image.height * self.scale}...")
        
        # Konwersja PIL -> OpenCV (BGR)
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        try:
            # Wysokiej jakości interpolacja Lanczos4 w OpenCV
            new_width = int(cv_img.shape[1] * self.scale)
            new_height = int(cv_img.shape[0] * self.scale)
            output = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Lekkie wyostrzenie po upscalingu (unsharp mask), aby tekstura wyglądała na HD
            gaussian_3 = cv2.GaussianBlur(output, (0, 0), 2.0)
            output = cv2.addWeighted(output, 1.5, gaussian_3, -0.5, 0)
            
            # Konwersja z powrotem na PIL (RGB)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output_rgb)
        except Exception as e:
            print(f"  [Upscaler Błąd] {e}. Wracam do skalowania podstawowego (PIL).")
            new_size = (image.width * self.scale, image.height * self.scale)
            return image.resize(new_size, Image.LANCZOS)
