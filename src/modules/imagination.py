import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import chromadb
import json

class InternalKnowledgeGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Moduł pełniący rolę "Wewnętrznej Wyobraźni" (Internal Knowledge) systemu.
        Używany jako Fallback, gdy sieć internetowa nie posiada odpowiedniego zdjęcia, 
        lub gdy DuckDuckGo zablokuje zapytania.
        Model ten korzysta ze swojej wewnętrznej pamięci (wag sieci neuronowej trenowanych 
        na miliardach obrazów) do "wyobrażenia sobie" jak powinien wyglądać dany obiekt.
        """
        print("Inicjalizacja modułu Wewnętrznej Wyobraźni (Text-to-Image Fallback)...")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
        # Inicjalizacja "Wizualnej Bazy Wiedzy" do zapisywania doświadczeń
        self.chroma_client = chromadb.PersistentClient(path="chroma_db_visual")
        self.collection = self.chroma_client.get_or_create_collection(name="visual_knowledge")

    def _load_model(self):
        if self.pipe is None:
            print("  [Imagination] Ładowanie modelu Text-to-Image do VRAM...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.pipe.safety_checker = None
            self.pipe.set_progress_bar_config(disable=True)
            print("  [Imagination] Model załadowany.")

    def remember_visual_features(self, prompt: str, features: dict):
        """
        Zapisuje zczytane cechy wizualne (teksturę, materiał, kształt, kolory) 
        do bazy wektorowej, budując własną bibliotekę natury i przedmiotów.
        """
        try:
            # Upewnienie się, że features to słownik składający się z tekstów
            clean_features = {}
            for k, v in features.items():
                if isinstance(v, list):
                    clean_features[k] = ", ".join(map(str, v))
                elif isinstance(v, str):
                    clean_features[k] = v
                else:
                    clean_features[k] = str(v)

            if not clean_features:
                return

            self.collection.add(
                documents=[prompt],
                metadatas=[clean_features],
                ids=[prompt.replace(" ", "_").lower()]
            )
            print(f"  [Imagination] Zapisano do pamięci wizualnej cechy dla: '{prompt}'")
        except Exception as e:
            # Prawdopodobnie już istnieje w bazie, możemy go zaktualizować (upsert)
            try:
                self.collection.upsert(
                    documents=[prompt],
                    metadatas=[clean_features],
                    ids=[prompt.replace(" ", "_").lower()]
                )
                print(f"  [Imagination] Zaktualizowano w pamięci wizualnej cechy dla: '{prompt}'")
            except Exception as e2:
                print(f"  [Imagination Błąd] Nie udało się zapisać cech: {e2}")

    def recall_visual_features(self, prompt: str) -> str:
        """
        Przeszukuje "Wizualną Bazę Wiedzy", by przypomnieć sobie jak zazwyczaj wygląda dany obiekt.
        Na podstawie zapamiętanych wektorów i pikseli (jako metadane) składa bogaty prompt.
        """
        try:
            results = self.collection.query(query_texts=[prompt], n_results=1)
            if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                meta = results['metadatas'][0][0]
                recalled = []
                
                if 'colors' in meta: recalled.append(f"dominant colors: {meta['colors']}")
                if 'texture' in meta: recalled.append(f"texture: {meta['texture']}")
                if 'material_or_surface' in meta: recalled.append(f"made of {meta['material_or_surface']}")
                if 'shape' in meta: recalled.append(f"shape: {meta['shape']}")
                if 'specific_features' in meta: recalled.append(f"features: {meta['specific_features']}")
                if 'size_category' in meta: recalled.append(f"size: {meta['size_category']}")
                
                # Dodatkowy fallback na stare klucze
                if not recalled:
                    if 'texture' in meta: recalled.append(meta['texture'])
                    if 'material' in meta: recalled.append(meta['material'])
                    if 'shape' in meta: recalled.append(meta['shape'])
                
                if recalled:
                    mem = ", ".join(recalled)
                    print(f"  [Imagination] Znalazłem we własnej bazie wiedzy informacje o '{prompt}': {mem}")
                    return mem
        except Exception as e:
            print(f"  [Imagination] Błąd podczas przypominania: {e}")
        return ""

    def imagine_object(self, prompt: str, style: str) -> Image.Image:
        """
        "Wyobraża sobie" obiekt od zera na białym tle, bazując na prompt'cie i zgromadzonej wiedzy.
        """
        self._load_model()
        
        # Próba przypomnienia sobie detali wizualnych z poprzednich doświadczeń
        memory_modifier = self.recall_visual_features(prompt)
        
        # Konstruowanie promptu zmuszającego sieć do stworzenia obiektu izolowanego
        imagination_prompt = f"A single perfectly isolated {prompt}, {memory_modifier}, centered, front view, solid white background, high quality concept art, 3d render"
        if "Low Poly" in style:
            imagination_prompt += ", low poly"
        elif "Voxel" in style:
            imagination_prompt += ", voxel style"
            
        print(f"  [Imagination] Generowanie z własnej 'wyobraźni': '{imagination_prompt}'")
        
        try:
            image = self.pipe(
                prompt=imagination_prompt,
                negative_prompt="background details, environment, scenery, multiple objects, text, watermark, blurry, shadows on background",
                num_inference_steps=25,
                guidance_scale=8.0
            ).images[0]
            
            # Zwolnienie VRAM
            if self.device == "cuda":
                self.pipe.to("cpu")
                torch.cuda.empty_cache()
                
            return image
        except Exception as e:
            print(f"  [Imagination Błąd] Sieć nie potrafiła wyobrazić sobie obiektu: {e}")
            return None
