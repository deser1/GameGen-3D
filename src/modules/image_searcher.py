import os
import requests
import numpy as np
import cv2
import time
import random
from io import BytesIO
from PIL import Image
from ddgs import DDGS
from rembg import remove

# Słownik podstawowych kolorów w przestrzeni HSV (Hue, Saturation, Value)
# Zakres Hue w OpenCV to 0-179, Saturation 0-255, Value 0-255
COLOR_RANGES = {
    'red': [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [180, 255, 255])], # Czerwony zawija się na osi Hue
    'green': [([35, 70, 50], [85, 255, 255])],
    'blue': [([90, 70, 50], [130, 255, 255])],
    'yellow': [([20, 70, 50], [35, 255, 255])],
    'orange': [([10, 70, 50], [20, 255, 255])],
    'purple': [([125, 70, 50], [150, 255, 255])],
    'pink': [([140, 70, 50], [170, 255, 255])],
    'brown': [([10, 50, 20], [20, 255, 200])],
    'black': [([0, 0, 0], [180, 255, 40])],
    'white': [([0, 0, 200], [180, 30, 255])],
    'gray': [([0, 0, 40], [180, 30, 200])]
}

class ImageReferenceSearcher:
    def __init__(self):
        """
        Inicjalizuje moduł wyszukiwania obrazów referencyjnych w Internecie.
        Wykorzystuje wyszukiwarkę DuckDuckGo, aby nie wymagać kluczy API.
        """
        print("[Init] Ładowanie modułu ImageReferenceSearcher (DuckDuckGo)")

    def _extract_colors_from_prompt(self, prompt: str) -> list[str]:
        """Znajduje słowa kluczowe kolorów w prompcie."""
        prompt_lower = prompt.lower()
        found_colors = []
        for color in COLOR_RANGES.keys():
            # Sprawdzanie również polskich odpowiedników
            pl_colors = {
                'red': ['czerwon'], 'green': ['zielon'], 'blue': ['niebiesk'],
                'yellow': ['żółt', 'zolt'], 'orange': ['pomarańcz'], 
                'purple': ['fiolet'], 'pink': ['różow', 'rozow'],
                'brown': ['brąz', 'braz', 'dęb', 'drewn'], 'black': ['czarn'], 
                'white': ['biał', 'bial'], 'gray': ['szar', 'żelaz', 'metal']
            }
            
            if color in prompt_lower:
                found_colors.append(color)
            else:
                for pl_word in pl_colors.get(color, []):
                    if pl_word in prompt_lower:
                        found_colors.append(color)
                        break
        return list(set(found_colors))

    def _analyze_image_color(self, image: Image.Image, target_colors: list[str]) -> bool:
        """
        Sprawdza, czy obraz zawiera odpowiednią ilość docelowych kolorów.
        Jeśli w prompcie nie ma kolorów, zwraca True.
        """
        if not target_colors:
            return True
            
        # Konwersja z PIL na OpenCV format (BGR -> HSV)
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        
        # Obliczamy łączny procent docelowych kolorów na obrazku
        total_matched_pixels = 0
        
        for color in target_colors:
            color_mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)
            for (lower, upper) in COLOR_RANGES[color]:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv_img, lower_np, upper_np)
                color_mask = cv2.bitwise_or(color_mask, mask)
                
            total_matched_pixels += cv2.countNonZero(color_mask)
            
        match_percentage = (total_matched_pixels / total_pixels) * 100
        
        print(f"    -> Analiza kolorów: znaleziono {match_percentage:.1f}% pikseli pasujących do {target_colors}")
        
        # Jeśli co najmniej 15% obrazka pasuje do wymaganego koloru, uznajemy to za sukces
        return match_percentage > 15.0

    def _download_image(self, url: str) -> Image.Image:
        """Pobiera obraz z podanego adresu URL."""
        try:
            # Udajemy przeglądarkę, żeby ominąć blokady (np. Cloudflare, błędy 403 Forbidden)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            if response.status_code == 200:
                img = Image.open(response.raw).convert("RGB")
                return img
            else:
                print(f"  [Searcher Ostrzeżenie] Nie udało się pobrać zdjęcia z URL. Kod błędu HTTP: {response.status_code}")
                return None
        except Exception as e:
            # Zignorowanie logowania błędów timeout dla każdego zepsutego linku, by nie śmiecić konsoli
            return None

    def _verify_single_object(self, image: Image.Image) -> bool:
        """
        Analizuje obraz (z usuniętym tłem, format RGBA) pod kątem ilości wyraźnych obiektów.
        Używa kanału Alpha i detekcji konturów OpenCV.
        Zwraca True, jeśli na zdjęciu znajduje się tylko jeden główny obiekt.
        """
        # Jeśli obraz nie ma kanału Alpha (nie usunięto z niego tła), nie możemy wiarygodnie policzyć wysp
        if image.mode != "RGBA":
            return True

        # Wyciągnięcie kanału Alpha (przezroczystości) z obrazu
        img_array = np.array(image)
        alpha_channel = img_array[:, :, 3]

        # Binaryzacja (progowanie) - upewniamy się, że to czarno-biała maska
        _, thresh = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)

        # Usunięcie szumu (drobnych plamek) za pomocą operacji morfologicznych
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Znalezienie konturów (zewnętrznych)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrowanie zbyt małych konturów (np. malutkich odłamków pozostałych po Rembg)
        total_image_area = alpha_channel.shape[0] * alpha_channel.shape[1]
        min_contour_area = total_image_area * 0.05 # Kontur musi zajmować min. 5% obrazu, by uznać go za osobny obiekt

        significant_objects = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                significant_objects += 1

        if significant_objects == 0:
            print("    -> [Contour] Nie wykryto żadnego głównego obiektu na zdjęciu po wycięciu tła.")
            return False
        elif significant_objects > 1:
            print(f"    -> [Contour] Wykryto {significant_objects} osobne obiekty! Zdjęcie odrzucone (wymagany jest dokładnie jeden obiekt).")
            return False
            
        print("    -> [Contour] Wykryto dokładnie jeden główny obiekt. Zdjęcie zatwierdzone.")
        return True

    def search_reference(self, prompt: str, max_results: int = 1) -> Image.Image:
        """
        Wyszukuje obrazy pasujące do promptu, weryfikuje kolorystykę,
        usuwa z nich tło za pomocą rembg i weryfikuje czy na zdjęciu znajduje się tylko jeden obiekt.
        Zwraca czysty obiekt. Posiada wbudowany timeout na wypadek braku internetu.
        """
        search_query = f"{prompt} 3d model concept art white background isolated"
        print(f"[Searcher] Wyszukiwanie referencji w sieci: '{search_query}'")
        
        # Wyciągamy kolory z promptu
        target_colors = self._extract_colors_from_prompt(prompt)
        if target_colors:
            print(f"  [Searcher] Wymagane kolory do analizy wizyjnej: {target_colors}")
        
        max_retries = 2 # Zmniejszone z 3, aby szybciej przechodzić do wyobraźni w razie awarii sieci
        for attempt in range(max_retries):
            try:
                # Ograniczamy czas wykonania zapytania do DuckDuckGo za pomocą własnego proxy timeout
                # Ustawiamy timeout globalny na 10 sekund dla całej wyszukiwarki (jeśli DDGS by się zawiesiło)
                with DDGS(timeout=10) as ddgs:
                    # Pobieramy więcej wyników, bo niektóre mogą odpaść na weryfikacji koloru lub ilości obiektów
                    results_gen = ddgs.images(search_query, max_results=max_results + 10)
                    results = list(results_gen)
                    
                    if not results:
                        print("  [Searcher] Brak wyników z wyszukiwarki. Przechodzę do wyobraźni.")
                        return None
                    
                    for res in results:
                        image_url = res.get('image')
                        if image_url:
                            print(f"  -> Próba pobrania i weryfikacji referencji z: {image_url}")
                            img = self._download_image(image_url)
                            
                            if img is not None:
                                # AI wizyjnie sprawdza obraz pod kątem kolorów
                                if self._analyze_image_color(img, target_colors):
                                    print("  [Searcher] Znaleziono obraz referencyjny o odpowiedniej kolorystyce.")
                                    print("  [Searcher] Usuwanie tła (Rembg)...")
                                    # Usuwamy tło
                                    img_no_bg = remove(img)
                                    
                                    # Weryfikacja za pomocą OpenCV czy po wycięciu tła mamy tylko JEDEN główny obiekt
                                    if self._verify_single_object(img_no_bg):
                                        print("  [Searcher] Sukces! Zdjęcie spełnia wszystkie rygorystyczne kryteria (Kolor + 1 Obiekt).")
                                        return img_no_bg
                                    else:
                                        print("  [Searcher] Odrzucono obraz: Wykryto wiele obiektów (lub zero). Szukam dalej...")
                                else:
                                    print("  [Searcher] Odrzucono obraz: Niewystarczające dopasowanie kolorów. Szukam dalej...")
                                    
                print("  [Searcher] Ostrzeżenie: Nie znaleziono obrazu spełniającego wszystkie kryteria.")
                return None
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"[Searcher] Błąd wyszukiwania (Próba {attempt+1}/{max_retries}): {e}")
                
                # Jeśli błąd to timeout sieciowy, brak internetu lub DNS resolution failed, od razu przerywamy pętlę i przechodzimy do wyobraźni
                if any(kw in error_msg for kw in ["timeout", "connection", "network is unreachable", "name resolution", "read timeout"]):
                    print("  [Searcher] Poważny problem z siecią/Internetem. Natychmiastowe przerwanie i przejście do Wewnętrznej Wyobraźni AI.")
                    return None
                    
                if "403 ratelimit" in error_msg or "ratelimit" in error_msg:
                    if attempt < max_retries - 1:
                        sleep_time = random.uniform(2.0, 4.0)
                        print(f"  [Searcher] DuckDuckGo nałożyło limit. Czekam {sleep_time:.1f} sekund...")
                        time.sleep(sleep_time)
                    else:
                        print("  [Searcher] Przekroczono limit prób DuckDuckGo.")
                        return None
                else:
                    # Inny nieznany błąd - lepiej zrezygnować by nie zawieszać systemu
                    return None
                    
        return None
