from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

class TextToMultiViewGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Inicjalizuje model dyfuzyjny z HuggingFace w trybie ControlNet + Image-to-Image.
        Zamiast generować obraz od zera na podstawie tekstu, używamy 
        znalezionego w sieci obrazu referencyjnego, aby zachować strukturę (dzięki krawędziom Canny) 
        i wygląd (dzięki img2img) przed wygenerowaniem widoków bocznych i tylnych.
        """
        self.model_id = model_id
        print(f"[Init] Ładowanie modelu dyfuzyjnego z HuggingFace (ControlNet Img2Img): {model_id}")
        
        # Sprawdzamy czy dostępne jest GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] Używane urządzenie obliczeniowe: {self.device}")
        
        try:
            # Ładujemy ControlNet dla wykrywania krawędzi (Canny)
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.model_id, 
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            self.pipeline = self.pipeline.to(self.device)
            # Wyłączenie weryfikacji bezpieczeństwa by przyspieszyć działanie
            self.pipeline.set_progress_bar_config(disable=False)
        except Exception as e:
            print(f"[Błąd] Nie udało się załadować modelu {model_id} z ControlNetem. Błąd: {e}")
            self.pipeline = None

    def enhance_prompt(self, prompt: str, style: str = "Fotorealistyczny (PBR)") -> str:
        """
        Wzbogaca krótki prompt użytkownika o detale wizualne pod kątem wybranego stylu.
        """
        base_prompt = f"{prompt}, high quality 3d asset, white background, centered, 4k resolution"
        
        # Wstrzykiwanie stylu do modelu dyfuzyjnego
        if "Low Poly" in style:
            enhanced = f"{base_prompt}, low poly 3d model, flat shading, geometric shapes, minimal textures, sharp edges"
        elif "Voxel" in style:
            enhanced = f"{base_prompt}, voxel art style, made of cubes, minecraft style, blocky, retro 8-bit 3d"
        elif "Cyberpunk" in style:
            enhanced = f"{base_prompt}, cyberpunk 2077 style, sci-fi, neon lights, metallic parts, futuristic, detailed machinery"
        elif "Anime" in style:
            enhanced = f"{base_prompt}, anime art style, cel shaded, flat colors, studio ghibli style, drawn outlines"
        else:
            # Domyślny PBR
            enhanced = f"{base_prompt}, physically based rendering, highly detailed, realistic materials, octane render"
            
        print(f"  [Prompt] Zastosowano styl ({style}). Wzbogacony prompt: '{enhanced}'")
        return enhanced

    def generate(self, prompt: str, reference_image: Image.Image = None, style: str = "Fotorealistyczny (PBR)", progress_callback=None) -> list[Image.Image]:
        """
        Generuje widoki korzystając z modelu Image-to-Image wspomaganego ControlNetem (Canny).
        Dzięki temu bryła przedmiotu nie zmieni nagle swojego ogólnego kształtu na widoku z boku.
        """
        enhanced_prompt = self.enhance_prompt(prompt, style)
        print(f"[Generate] Uruchamianie Stable Diffusion (ControlNet) dla: '{prompt}' w stylu {style}...")
        
        generated_images = []
        views = ["front view", "side view", "back view", "top view"]
        
        if self.pipeline is not None:
            if reference_image:
                if progress_callback:
                    progress_callback(0.26, "Przygotowanie mapy kształtu (Canny Edge)...")
                    
                # Zamiana przezroczystego tła z Rembg na czyste białe tło dla SD
                if reference_image.mode == "RGBA":
                    white_bg = Image.new("RGB", reference_image.size, (255, 255, 255))
                    white_bg.paste(reference_image, mask=reference_image.split()[3])
                    ref_img = white_bg.resize((512, 512))
                else:
                    ref_img = reference_image.resize((512, 512))
                    
                # Wyciągamy krawędzie do ControlNetu za pomocą OpenCV Canny
                cv_img = np.array(ref_img)
                edges = cv2.Canny(cv_img, 100, 200)
                edges_3c = np.stack([edges]*3, axis=-1)
                control_img = Image.fromarray(edges_3c)
            else:
                # Fallback do szumu i braku kontroli jeśli nie ma obrazka
                ref_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                control_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
                
            for i, view in enumerate(views):
                view_prompt = f"{enhanced_prompt}, {view}, white background, centered"
                print(f"  -> Przygotowywanie widoku: {view}")
                
                if progress_callback:
                    # Rozkładamy postęp na 4 widoki (od 0.27 do 0.39)
                    pct = 0.27 + (i * 0.03)
                    progress_callback(pct, f"Generowanie widoku AI: {view}...")

                # Używamy ControlNetu, by kształt (sylwetka) z pierwszego zdjęcia 
                # został zachowany przy generowaniu widoków bocznych (strength = wierność kolorom, controlnet = wierność kształtowi)
                strength = 0.4 if i == 0 and reference_image else 0.85
                controlnet_conditioning_scale = 1.0 if i == 0 else 0.6 # Słabszy wpływ krawędzi z przodu na widok z boku

                # Niestety nie możemy bezpośrednio podmienić desc w domyślnym pasku pipeline.
                # Wyłączymy domyślny pasek diffusers i zrobimy po prostu własny napis lub własny callback
                self.pipeline.set_progress_bar_config(disable=True)
                
                print(f"     [SD] Rysowanie detali ({view}) przez sieć neuronową...")
                image = self.pipeline(
                    prompt=view_prompt,
                    image=ref_img,
                    control_image=control_img,
                    strength=strength,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                generated_images.append(image)
                
            print("[Generate] Pomyślnie wygenerowano widoki przez AI.")
        else:
            print("[Ostrzeżenie] Model dyfuzyjny nie załadował się. Fallback do atrap obrazów.")
            for i, view in enumerate(views):
                img_array = np.random.randint(100, 150, (512, 512, 3), dtype=np.uint8)
                generated_images.append(Image.fromarray(img_array))
            
        return generated_images
