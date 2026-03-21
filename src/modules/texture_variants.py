import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

class TextureVariantGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Moduł Procedural Wear / Texture Variations.
        Używa modelu Image2Image (np. Stable Diffusion), aby na bazie oryginalnej
        tekstury Albedo wygenerować jej warianty (np. ośnieżona, zardzewiała, porośnięta mchem),
        zachowując jej układ UV.
        """
        print("Inicjalizacja modułu Wariantów Tekstur (Procedural Wear)...")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
    def _load_model(self):
        if self.pipe is None:
            print("  [TextureVariants] Ładowanie modelu SD Img2Img do VRAM...")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            # Wyłączenie safety checkera dla szybkości i unikania fałszywych alarmów przy np. "bloody"
            self.pipe.safety_checker = None
            print("  [TextureVariants] Model załadowany.")

    def generate_variants(self, base_image: Image.Image, item_name: str, variants: list[str] = ["snowy", "mossy", "burnt", "bloody"], output_dir: str = "output/textures") -> dict:
        """
        Generuje warianty podanej tekstury Albedo.
        """
        self._load_model()
        
        # Konwersja obrazu do RGB jeśli to RGBA
        if base_image.mode == "RGBA":
            # Tworzymy białe tło dla przezroczystości (niezbędne dla SD)
            background = Image.new("RGB", base_image.size, (255, 255, 255))
            background.paste(base_image, mask=base_image.split()[3])
            base_image = background
        elif base_image.mode != "RGB":
            base_image = base_image.convert("RGB")
            
        # Zmniejszenie rozmiaru na czas generowania dla szybkości
        original_size = base_image.size
        base_image = base_image.resize((512, 512))
            
        results = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for var in variants:
            print(f"  [TextureVariants] Generowanie wariantu: '{var}' dla '{item_name}'...")
            prompt = f"{var} {item_name}, highly detailed texture, 4k resolution, pbr material, flat lighting"
            negative_prompt = "shadows, highlights, 3d render, background, out of focus, blurry"
            
            try:
                self.pipe.set_progress_bar_config(disable=True)
                print(f"     [SD Img2Img] Przetwarzanie tekstury dla wariantu ({var})...")
                
                # strength określa jak bardzo zmieniamy obraz. 0.6-0.75 jest dobre do zachowania kształtu
                output_img = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=base_image,
                    strength=0.65,
                    guidance_scale=7.5,
                    num_inference_steps=20
                ).images[0]
                
                # Powrót do oryginalnego rozmiaru
                output_img = output_img.resize(original_size, Image.Resampling.LANCZOS)
                
                path = os.path.join(output_dir, f"variant_{var}.png")
                output_img.save(path)
                results[var] = path
                print(f"  [TextureVariants] Zapisano wariant: {path}")
                
            except Exception as e:
                print(f"  [TextureVariants Błąd] Nie udało się wygenerować wariantu {var}: {e}")
                
        # Zwolnienie pamięci VRAM
        if self.device == "cuda":
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
            
        return results
