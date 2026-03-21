import os
import torch
import scipy.io.wavfile as wavfile
from diffusers import AudioLDMPipeline

class SFXGenerator:
    def __init__(self, model_id: str = "cvssp/audioldm-s-2-v2"):
        """
        Moduł generatora efektów dźwiękowych (SFX) na podstawie tekstu.
        Używa małego modelu AudioLDM, aby wygenerować np. dźwięk uderzenia w obiekt.
        """
        print("Inicjalizacja modułu SFX Generator (AudioLDM)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_id = model_id
        # Inicjalizacja opóźniona (lazy loading), by nie blokować VRAM na starcie
        
    def _load_model(self):
        if self.pipe is None:
            print("  [SFX] Ładowanie modelu AudioLDM do VRAM...")
            self.pipe = AudioLDMPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
            self.pipe = self.pipe.to(self.device)
            print("  [SFX] Model załadowany.")

    def generate_sfx(self, prompt: str, output_path: str = "output/sfx/item_sound.wav", material: str = "") -> str:
        """
        Generuje krótki dźwięk pasujący do przedmiotu (np. dźwięk upuszczenia, uderzenia).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self._load_model()
        
        # Budowanie promptu dla audio (po angielsku, bo model lepiej tak działa)
        # Przykład: "Sound of a heavy wooden barrel dropping on the ground"
        audio_prompt = f"Sound effect of dropping a {prompt} made of {material}, high quality, game foley"
        
        print(f"  [SFX] Generowanie dźwięku dla: '{audio_prompt}'...")
        
        try:
            # Generowanie 2 sekundowego dźwięku (num_inference_steps=10 by przyspieszyć)
            audio = self.pipe(audio_prompt, num_inference_steps=15, audio_length_in_s=2.0).audios[0]
            
            # Zapis do pliku WAV
            sample_rate = 16000
            wavfile.write(output_path, rate=sample_rate, data=audio)
            print(f"  [SFX] Dźwięk wygenerowany: {output_path}")
            
            # Zwolnienie VRAM, ponieważ główny model 3D go potrzebuje
            if self.device == "cuda":
                self.pipe.to("cpu")
                torch.cuda.empty_cache()
                
            return output_path
        except Exception as e:
            print(f"  [SFX Błąd] Nie udało się wygenerować dźwięku: {e}")
            return None
