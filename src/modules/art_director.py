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

    def extract_hu_moments(self, image: Image.Image) -> list:
        """
        Wyciąga 7 momentów Hu (niezmienników kształtu) z obrazka.
        Służą one do matematycznego porównywania konturów obiektów, niezależnie od skali czy obrotu.
        """
        try:
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            # Obraz ma zazwyczaj białe tło - odwracamy kolory do znalezienia konturów (obiekt biały, tło czarne)
            _, thresh = cv2.threshold(cv_img, 240, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
                
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments['m00'] == 0:
                return []
                
            hu_moments = cv2.HuMoments(moments).flatten().tolist()
            
            # Konwersja na skalę logarytmiczną dla lepszego i stabilnego porównywania wartości
            log_hu = []
            for h in hu_moments:
                if h == 0:
                    log_hu.append(0.0)
                else:
                    log_hu.append(-1 * np.copysign(1.0, h) * np.log10(abs(h)))
            return log_hu
        except Exception as e:
            print(f"  [Art Director] Błąd podczas ekstrakcji momentów Hu: {e}")
            return []

    def match_shapes(self, image: Image.Image, remembered_log_hu: list) -> float:
        """
        Porównuje matematyczny kształt na obrazku z zapamiętanymi momentami Hu.
        Zwraca dystans (im mniejszy, tym bardziej podobne kształty).
        """
        current_log_hu = self.extract_hu_moments(image)
        if not current_log_hu or not remembered_log_hu:
            return 0.0 # Brak danych do porównania
            
        # Dystans L1 (suma różnic bezwzględnych poszczególnych momentów Hu)
        dist = sum(abs(a - b) for a, b in zip(current_log_hu, remembered_log_hu))
        return dist

    def extract_lasso_points(self, image: Image.Image, num_points: int = 64) -> list:
        """
        Tworzy 'lasso' - wektorowy obrys kształtu składający się z N znormalizowanych punktów.
        Pozwala na bezpośrednie uczenie się fizycznego kształtu obiektu.
        """
        try:
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(cv_img, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return []
                
            contour = max(contours, key=cv2.contourArea)
            pts = contour.reshape(-1, 2)
            if len(pts) < 3:
                return []
                
            # Obliczanie długości obwodu (lasso)
            diffs = np.diff(pts, axis=0)
            diffs = np.vstack([diffs, pts[0] - pts[-1]])
            dists = np.linalg.norm(diffs, axis=1)
            cum_dists = np.concatenate(([0], np.cumsum(dists)))
            total_len = cum_dists[-1]
            if total_len == 0:
                return []
                
            target_dists = np.linspace(0, total_len, num_points, endpoint=False)
            resampled = np.zeros((num_points, 2))
            
            for i, td in enumerate(target_dists):
                idx = np.searchsorted(cum_dists, td, side='right') - 1
                if idx < 0: idx = 0
                if idx >= len(pts): idx = len(pts) - 1
                
                p1 = pts[idx]
                p2 = pts[(idx + 1) % len(pts)]
                
                segment_len = dists[idx]
                if segment_len == 0:
                    resampled[i] = p1
                else:
                    t = (td - cum_dists[idx]) / segment_len
                    resampled[i] = p1 + t * (p2 - p1)
                    
            # Normalizacja punktów wektorowych (centrowanie i skalowanie do -1...1)
            centroid = np.mean(resampled, axis=0)
            resampled -= centroid
            max_dist = np.max(np.linalg.norm(resampled, axis=1))
            if max_dist > 0:
                resampled /= max_dist
                
            return resampled.flatten().tolist()
        except Exception as e:
            print(f"  [Art Director] Błąd podczas ekstrakcji wektorów lassa: {e}")
            return []

    def match_lasso_shapes(self, image: Image.Image, remembered_lasso: list) -> float:
        """
        Porównuje punkty wektorowe 'lassa' z zapamiętanym kształtem.
        Zwraca średni błąd kwadratowy (MSE) z uwzględnieniem najlepszego dopasowania punktu startowego.
        """
        current_lasso = self.extract_lasso_points(image)
        if not current_lasso or not remembered_lasso or len(current_lasso) != len(remembered_lasso):
            return 0.0
            
        num_points = len(current_lasso) // 2
        curr_pts = [(current_lasso[2*i], current_lasso[2*i+1]) for i in range(num_points)]
        rem_pts = [(remembered_lasso[2*i], remembered_lasso[2*i+1]) for i in range(num_points)]
        
        min_mse = float('inf')
        
        # Sprawdzanie dla każdego możliwego punktu startowego na konturze
        for shift in range(num_points):
            shifted_curr = curr_pts[shift:] + curr_pts[:shift]
            mse = sum((c[0]-r[0])**2 + (c[1]-r[1])**2 for c, r in zip(shifted_curr, rem_pts)) / num_points
            if mse < min_mse:
                min_mse = mse
                
        # Sprawdzanie również odbicia lustrzanego (reversed)
        curr_pts_rev = curr_pts[::-1]
        for shift in range(num_points):
            shifted_curr = curr_pts_rev[shift:] + curr_pts_rev[:shift]
            mse = sum((c[0]-r[0])**2 + (c[1]-r[1])**2 for c, r in zip(shifted_curr, rem_pts)) / num_points
            if mse < min_mse:
                min_mse = mse

        return min_mse

    def visualize_lasso(self, lasso_points: list, size: int = 512) -> Image.Image:
        """
        Rysuje wizualizację punktów wektorowych Lassa do podglądu na stronie WWW.
        """
        if not lasso_points:
            return None
        try:
            img = np.zeros((size, size, 3), dtype=np.uint8)
            pts = np.array(lasso_points).reshape(-1, 2)
            
            # Punkty są znormalizowane w przedziale ok. -1 do 1. Skalujemy je do rozmiaru obrazka.
            scale = (size / 2) * 0.8
            pts = pts * scale + (size / 2)
            pts = pts.astype(np.int32)
            
            # Rysowanie linii lassa i punktów kontrolnych
            for i in range(len(pts)):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i+1)%len(pts)])
                cv2.line(img, p1, p2, (0, 255, 0), 2)
                cv2.circle(img, p1, 4, (0, 0, 255), -1)
                
            return Image.fromarray(img)
        except Exception as e:
            print(f"  [Art Director Błąd] Wizualizacja lassa: {e}")
            return None

    def is_humanoid(self, prompt: str, image: Image.Image) -> bool:
        """
        Ocenia przy użyciu VLM, czy obiekt jest humanoidalny (wymaga kości i auto-riggingu).
        """
        fallback_check = any(word in prompt.lower() for word in ["postać", "potwór", "ludzik", "zwierzę", "człowiek", "character", "monster", "animal", "human"])
        if not self.available:
            return fallback_check
            
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            system_prompt = (
                "Jesteś analitykiem 3D. Oceniasz, czy podany obiekt wymaga szkieletu (rig/kości) do animacji. "
                "Odpowiedz w formacie JSON z polem 'is_humanoid': true lub false. "
                "Ustaw true jeśli to żywa istota (człowiek, zwierzę, potwór, robot z kończynami). "
                "Ustaw false jeśli to obiekt nieożywiony, rekwizyt lub otoczenie (beczka, broń, roślina, stół)."
            )

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Opis: '{prompt}'. Czy to wymaga szkieletu postaci?", 'images': [img_bytes]}
                ],
                format='json'
            )
            import json
            result = json.loads(response['message']['content'])
            is_hum = result.get('is_humanoid', fallback_check)
            print(f"  [Art Director] Wykryto potrzebę Auto-Riggingu (Humanoid): {is_hum}")
            return is_hum
        except Exception as e:
            print(f"  [Art Director Błąd] Klasyfikacja humanoida: {e}")
            return fallback_check

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
            
            # 3. Ekstrakcja matematycznego obrysu / kształtu (momenty Hu z OpenCV)
            hu_moments = self.extract_hu_moments(image)
            if hu_moments:
                traits['hu_moments'] = hu_moments
                
            # 4. Ekstrakcja obrysu wektorowego "Lasso"
            lasso_points = self.extract_lasso_points(image)
            if lasso_points:
                traits['lasso_points'] = lasso_points
                
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
