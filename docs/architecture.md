# Architektura Systemu GameGen-3D

GameGen-3D to kompleksowy, w pełni zautomatyzowany rurociąg (pipeline) oparty na uczeniu maszynowym, służący do generowania gotowych modeli 3D, materiałów PBR, siatek kolizji oraz statystyk dla gier komputerowych na podstawie samych opisów tekstowych.

## Fazy Przetwarzania (Pipeline)

Proces został rozbudowany do kilku zaawansowanych etapów:

### 1. Pamięć Semantyczna (Vector Cache)
* **Moduł:** `MemoryManager` (ChromaDB + SentenceTransformers).
* **Działanie:** System sprawdza wektorowe podobieństwo promptu. Jeśli podobny obiekt był już generowany, natychmiast zwraca go z pamięci, oszczędzając czas.

### 2. Rozszerzenie Kontekstu i Analiza Sieci (RAG)
* **Moduł:** `ImageReferenceSearcher` (DuckDuckGo + OpenCV + Rembg).
* **Działanie:** Sztuczna inteligencja przeszukuje Internet w poszukiwaniu odpowiedniego obrazu referencyjnego. Weryfikuje zgodność kolorów w przestrzeni HSV względem promptu i automatycznie usuwa tło (Rembg / U2-Net).

### 3. Pamięć Wizualna i Wewnętrzna Wyobraźnia (Zrozumienie Świata)
* **Moduły:** `ArtDirectorVLM` + `InternalKnowledgeGenerator` (OpenCV + LLaVA + ChromaDB + Stable Diffusion).
* **Działanie:** System aktywnie "uczy się" jak wyglądają przedmioty, rośliny i zwierzęta. 
  - **Uczenie (Ekstrakcja Cecha):** Przy pomocy OpenCV analizuje przestrzeń pikseli i wektory dominujących kolorów, a następnie używając LLaVA ekstrahuje cechy fizyczne (tekstura kory/sierści, kształt, materiał). Te informacje są trwale zapisywane w wektorowej bazie danych.
  - **Wyobraźnia (Fallback):** Gdy sieć nie znajdzie odpowiedniego zdjęcia referencyjnego, moduł odpytuje własną pamięć wizualną. Wzbogaca prompt o zapamiętane cechy i wektory (np. *"pamiętam, że to ma skrzydła, łuski i kolor RGB(150,20,20)"*), a następnie generuje idealny obraz bazowy "z wyobraźni" za pomocą Text-to-Image.

### 4. Generowanie Wielowidokowe z Kontrolą Kształtu
* **Moduł:** `TextToMultiViewGenerator` (Stable Diffusion v1.5 + ControlNet Canny).
* **Działanie:** Na bazie wyciętego lub "wyobrażonego" obrazu i promptu, model generuje zsynchronizowane widoki ortograficzne obiektu. ControlNet pilnuje, by sylwetka była ściśle zachowana, co eliminuje halucynacje 3D. System obsługuje również wstrzykiwanie stylów (np. Voxel, Low Poly).

### 5. Błyskawiczna Rekonstrukcja LRM
* **Moduł:** `MultiViewTo3DReconstructor` (TripoSR).
* **Działanie:** Wykorzystuje architekturę Large Reconstruction Model (TSR) do przewidzenia pola NeRF/Triplane z obrazu i błyskawicznie ekstrahuje surową gęstą siatkę w czasie krótszym niż sekunda.

### 6. Optymalizacja pod Silniki Gier (Mesh, UV, PBR, LOD, Collision, Rigging)
* **Moduł:** `GameMeshOptimizer` (Trimesh, Xatlas, OpenCV, V-HACD).
* **Działanie:** 
  1. **Retopologia:** Symulacja Quad-Remeshingu, decymacja gęstej siatki i wygładzanie Taubina. Generowanie poziomów detali (LOD0 - LOD3).
  2. **Rozkładanie UV:** Automatyczne tworzenie szwów (seams) za pomocą Xatlas.
  3. **Wypalanie PBR:** Generowanie tekstur Albedo, Proceduralnych Map Normalnych i Roughness (filtry Sobela w OpenCV).
  4. **Fizyka:** Automatyczna generacja wypukłych siatek kolizyjnych za pomocą V-HACD (Hierarchical Approximate Convex Decomposition).
  5. **Auto-Rigging:** Automatyczne generowanie szkieletu (kości) w formacie JSON dla obiektów i postaci, gotowe do animacji w silniku.
  6. **4D Animation:** Proceduralne generowanie Blend Shapes (Morph Targets) dla obiektów animowanych.

### 7. Generowanie Zaawansowanych Wariantów i Przestrzeni
* **Moduły:** `GaussianSplatter`, `TextureVariantGenerator`, `SceneGeneratorLLM`.
* **Działanie:**
  - **Splatting:** Konwersja mesha do postaci gęstej chmury punktów (.ply) naśladującej 3D Gaussian Splatting.
  - **Warianty:** Img2Img generujący proceduralne uszkodzenia (rdza, mech, śnieg).
  - **Sceny:** Spatial LLM rozmieszcza wygenerowane pojedyncze modele na jednej scenie bazując na logicznym ułożeniu przestrzennym.

### 8. Generowanie Logiki Gry (LLM)
* **Moduł:** `GameLogicGenerator` (Ollama - np. Llama 3).
* **Działanie:** Model językowy analizuje prompt i styl obiektu, zwracając ustrukturyzowany plik JSON zawierający statystyki przedmiotu (waga, rzadkość, typ, opcje craftingu) gotowy do użycia w silniku gry.

### 9. VLM Art Director (Ocena Jakości)
* **Moduł:** `ArtDirectorVLM` (Ollama - LLaVA).
* **Działanie:** Model wizyjno-językowy działa jako wirtualny dyrektor artystyczny. Ocenia wygenerowany obraz w skali 1-10 i weryfikuje, czy pasuje do pierwotnego promptu, zapewniając automatyczną kontrolę jakości.

### 10. Generowanie Dźwięku (SFX)
* **Moduł:** `SFXGenerator` (AudioLDM).
* **Działanie:** Automatyczne tworzenie efektów dźwiękowych w formacie `.wav` na podstawie materiału i typu wygenerowanego obiektu, by asset od razu posiadał swój "dźwięk" w grze.

### 11. Interfejs i API
* **Moduły:** `Gradio` (Web UI) oraz `FastAPI` (REST API).
* **Działanie:** Projekt udostępnia intuicyjny panel w przeglądarce (wyświetlający model, statystyki, oceny i dźwięk) oraz pełnoprawne API pozwalające podłączyć pipeline bezpośrednio do silników takich jak Unreal Engine czy Unity.

## Wymagania Sprzętowe
Zalecane jest uruchamianie tego rurociągu na kartach NVIDIA z minimum 12-16GB VRAM, dysku SSD (dla bazy ChromaDB) oraz posiadanie zainstalowanego serwera Ollama dla logiki tekstowej.
