import ollama
import base64
from io import BytesIO
from PIL import Image

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
