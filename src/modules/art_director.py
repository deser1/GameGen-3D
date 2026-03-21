import ollama
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

class ArtDirectorVLM:
    def __init__(self, model_name: str = "llava"):
        """
        Moduł Art Directora (VLM - Vision Language Model).
        Wykorzystuje lokalny model LLaVA (uruchomiony przez Ollama) do oceny 
        wygenerowanego widoku i podjęcia decyzji, czy model jest akceptowalny.
        """
        self.model_name = model_name
        self.available = False
        try:
            # Sprawdzenie czy model jest dostępny
            models = ollama.list()
            if any(m['name'].startswith(model_name) for m in models.get('models', [])):
                self.available = True
        except Exception:
            self.available = False

    def image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_visual_traits(self, prompt: str, image: Image.Image) -> dict:
        """
        Zczytuje cechy wizualne z obrazu (tekstura, kształt, kolory, rodzaj sierści/piór itp.)
        na podstawie analizy wektorów i pikseli obrazu oraz modelu VLM.
        Służy do budowania wewnętrznej bazy wiedzy o wyglądzie obiektów, zwierząt i natury.
        """
        if not self.available:
            return {}

        try:
            # 1. Analiza matematyczna (wektory/piksele) za pomocą OpenCV
            # Konwersja z PIL (RGB) do OpenCV (BGR)
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Ekstrakcja dominującego koloru (K-Means)
            pixels = cv_img.reshape((-1, 3))
            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Zamiana z BGR na RGB
            centers = np.uint8(centers)
            rgb_centers = [f"RGB({c[2]}, {c[1]}, {c[0]})" for c in centers]
            pixel_colors_str = ", ".join(rgb_centers)

            # 2. Analiza semantyczna (VLM)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            system_prompt = (
                "Jesteś systemem analitycznym AI do budowania bazy wiedzy o świecie. "
                "Twoim zadaniem jest przeanalizowanie podanego obrazka (który przedstawia obiekt/zwierzę/roślinę) "
                "oraz matematycznych danych o pikselach, a następnie wyekstrahowanie jego kluczowych cech wizualnych. "
                "Zwróć odpowiedź w ścisłym formacie JSON z polami (UWAGA: wartości w JSON muszą być po ANGIELSKU, "
                "aby można je było przekazać do generatora obrazów): "
                "'colors' (lista dominujących kolorów, uwzględnij dane z pikseli), "
                "'texture' (opis tekstury np. rough, smooth, bark, metallic), "
                "'shape' (ogólny kształt, np. oval, elongated, spreading), "
                "'material_or_surface' (z czego się składa: np. wood, fur, feathers, scales, leaves), "
                "'specific_features' (lista unikalnych cech, np. wings, tail, horns, branches), "
                "'size_category' (np. small, medium, large)."
            )

            user_message = (
                f"Zanalizuj ten obrazek, który przedstawia: '{prompt}'. \n"
                f"System wizyjny wyodrębnił z pikseli następujące wektory dominujących kolorów: {pixel_colors_str}. \n"
                f"Jakie ma cechy wizualne?"
            )

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {
                        'role': 'user', 
                        'content': user_message,
                        'images': [img_bytes]
                    }
                ],
                format='json'
            )
            
            import json
            traits = json.loads(response['message']['content'])
            print(f"  [Art Director] Wyekstrahowano cechy na podstawie pikseli: {list(traits.keys())}")
            return traits
            
        except Exception as e:
            print(f"  [Art Director Błąd] Błąd podczas ekstrakcji cech: {e}")
            return {}

    def review_model(self, prompt: str, image: Image.Image) -> dict:
        """
        Przekazuje obraz referencyjny i prompt do VLM w celu oceny jakości.
        Zwraca słownik z oceną i komentarzem.
        """
        if not self.available:
            return {"status": "skipped", "score": 10, "feedback": "VLM (LLaVA) niedostępny. Pomijam weryfikację."}

        try:
            # Konwersja obrazu PIL na bajty dla Ollama
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            system_prompt = (
                "Jesteś Dyrektorem Artystycznym w studiu gier (Art Director). "
                "Oceniasz czy wygenerowany model 3D (widoczny na obrazku) pasuje do opisu. "
                "Zwróć odpowiedź w formacie JSON z polami: "
                "'score' (liczba od 1 do 10, gdzie 10 to idealne dopasowanie), "
                "'approved' (true jeśli score >= 6, false w przeciwnym razie), "
                "'feedback' (krótki komentarz po polsku co jest dobrze a co źle)."
            )

            user_message = f"Opis przedmiotu: '{prompt}'. Oceń to."

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {
                        'role': 'user', 
                        'content': user_message,
                        'images': [img_bytes]
                    }
                ],
                format='json'
            )
            
            import json
            result = json.loads(response['message']['content'])
            print(f"  [Art Director] Ocena: {result.get('score')}/10. Zwerfyfikowano: {result.get('approved')}")
            return result
            
        except Exception as e:
            print(f"  [Art Director Błąd] {e}")
            return {"status": "error", "score": 10, "feedback": "Błąd podczas oceny."}
