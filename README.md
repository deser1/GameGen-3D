# GameGen-3D 🎮🤖

GameGen-3D to potężny, kompleksowy system sztucznej inteligencji służący do automatycznego generowania gotowych assetów do gier komputerowych (tzw. "AAA Ready"). Wystarczy wpisać tekstowy opis przedmiotu, a system zajmie się resztą.

## ✨ Kluczowe Funkcjonalności

1. **RAG & Wyszukiwanie Sieciowe (DuckDuckGo + Rembg):** AI wyszukuje referencje w internecie, analizuje kolory w HSV i automatycznie usuwa tło.
2. **Pamięć Wizualna i Wyobraźnia (OpenCV + LLaVA + ChromaDB + Stable Diffusion):** System "uczy się" wyglądu świata, analizując matematycznie piksele i wektory obrazów. Gdy nie ma internetu, AI potrafi z wyobraźni narysować własny obraz bazując na wyuczonych cechach fizycznych (wzbogacone prompty).
3. **Generowanie Multi-view (Stable Diffusion + ControlNet):** Zapewnia spójność kształtu i generuje zsynchronizowane widoki obiektu z każdej strony.
4. **Błyskawiczna Rekonstrukcja (TripoSR):** Architektura LRM zamienia widoki 2D w gęstą siatkę 3D w niecałą sekundę.
5. **Optymalizacja Gamedev (Trimesh + Xatlas + V-HACD):** System automatycznie decymuje siatkę, wygładza ją, wykonuje symulację Quad-Remeshingu, rozkłada profesjonalne mapy UV, wypala tekstury PBR (Albedo, Normal, Roughness) oraz tworzy siatki kolizji i poziomy detali (LOD).
6. **Auto-Rigging i Animacje 4D:** Generuje bazowy szkielet (kości) `.json` gotowy do importu w Unreal Engine/Unity oraz proceduralne Blend Shapes (Morph Targets).
7. **Gaussian Splatting i Warianty Tekstur:** System generuje chmury punktów `.ply` oraz alternatywne proceduralne materiały (np. ośnieżenie, zniszczenia) metodą Img2Img.
8. **Generowanie Scen:** Oparty na AI układ przestrzenny (Spatial LLM), pozwalający wygenerować całe kompozycje środowiska, składając kilka obiektów w spójną scenę `.glb`.
9. **Logika Gry (LLM - Llama 3 via Ollama):** Zwraca gotowe statystyki w JSON (waga, rzadkość, typ, materiał).
10. **Art Director (VLM - LLaVA):** Autonomiczny wirtualny dyrektor artystyczny ocenia wygenerowany widok przedmiotu w skali 1-10 i weryfikuje jego poprawność.
11. **Generator SFX (AudioLDM):** Generuje proceduralne efekty dźwiękowe pasujące do przedmiotu (np. dźwięk upadku drewna).

## 📂 Struktura Katalogów

* `src/` - Kod źródłowy projektu.
  * `pipeline.py` - Główny orkiestrator łączący 10 etapów w jeden rurociąg.
  * `modules/` - Poszczególne moduły (wyszukiwanie, AI, optymalizacja, LLM, VLM, SFX).
* `docs/` - Szczegółowa dokumentacja (`architecture.md`).
* `output/` - Wygenerowane modele `.glb`, kolizje, rigi, widoki `.png` oraz dźwięki `.wav`.
* `app.py` - Webowy interfejs użytkownika w Gradio.
* `api.py` - Serwer REST API (FastAPI) dla integracji z silnikami gier.

## 🛠️ Instalacja i Wymagania

Wymagany jest **Python 3.10+**, akceleracja **CUDA** (NVIDIA z min. 12-16GB VRAM) oraz zainstalowany lokalnie serwer **[Ollama](https://ollama.com/)** wraz z pobranymi modelami `llama3` i `llava`.

```bash
# Instalacja wymaganych bibliotek dla kart NVIDIA (CUDA)
pip install -r requirements.txt
```

### 🔴 Wsparcie dla kart AMD Radeon (Linux / ROCm)
Uruchomienie systemu na kartach AMD jest możliwe na systemach Linux przy użyciu platformy ROCm. Należy pominąć standardową instalację PyTorch z pliku `requirements.txt` i zainstalować wersję dedykowaną dla ROCm.

1. **Instalacja PyTorch dla AMD ROCm:**
```bash
# Zastąp wersję rocm6.0 najnowszą obsługiwaną przez Twoją dystrybucję
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```
2. **Instalacja reszty zależności:**
```bash
pip install diffusers transformers accelerate rembg gradio trimesh xatlas opencv-python duckduckgo-search chromadb
```
3. **Konfiguracja Ollama na AMD:**
Ollama natywnie wspiera karty Radeon na Linuksie. Wystarczy zainstalować standardową wersję, a modele językowe będą akcelerowane sprzętowo. Wymagane modele pobierzesz komendami:
```bash
ollama run llama3
ollama run llava
```
*Uwaga: Zależnie od środowiska, w plikach źródłowych `.py` wywołania sprawdzające `torch.cuda.is_available()` mogą wymagać mapowania zmiennych środowiskowych (np. `HSA_OVERRIDE_GFX_VERSION`), aby środowisko ROCm przedstawiło się PyTorchowi jako kompatybilne urządzenie obliczeniowe.*

## 🚀 Uruchomienie

Masz do wyboru dwa tryby pracy:

**1. Interfejs Graficzny (Web UI - Gradio):**
Uruchamia intuicyjny panel w przeglądarce, gdzie możesz wpisać prompt, wybrać styl wizualny i obejrzeć/posłuchać wygenerowanego modelu.
```bash
python app.py
```

**2. Serwer REST API (FastAPI):**
Uruchamia bezgłowy serwer gotowy na przyjmowanie zapytań POST z silnika gry (np. plugin w Unreal Engine).
```bash
python api.py
```

## 🔧 Rozwiązywanie problemów (Troubleshooting)

### Problem z kompilacją `torchmcubes` (Windows)
Standardowa implementacja TripoSR wymaga pakietu `torchmcubes`, który może być trudny do skompilowania na systemie Windows ze względu na zależności C++. 
Z tego powodu w naszym kodzie użyliśmy własnego mechanizmu **fallback**. 

Jeśli w przyszłości pobierzesz świeżą wersję oryginalnego repozytorium TripoSR, musisz zmodyfikować jej plik `tsr/models/isosurface.py`, aby korzystał z biblioteki `PyMCubes` (którą łatwo zainstalować przez pip: `pip install PyMCubes`).
Dodaj na samej górze pliku `isosurface.py`:
```python
try:
    from torchmcubes import marching_cubes
except ImportError:
    print("[TripoSR Ostrzeżenie] Nie znaleziono torchmcubes. Używam PyMCubes jako fallback.")
    import mcubes
    def marching_cubes(volume, isovalue):
        volume_np = volume.cpu().numpy()
        vertices, faces = mcubes.marching_cubes(volume_np, isovalue)
        vertices_tensor = torch.from_numpy(vertices.astype(np.float32)).to(volume.device)
        faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(volume.device)
        return vertices_tensor, faces_tensor
```
Zmniejsz także rozdzielczość siatki w wywołaniu `extract_mesh` (np. na `192` zamiast `256`), aby zapobiec zawieszaniu się procesu (bottleneck procesora). W naszym repozytorium ta zmiana jest już zaimplementowana w `src/modules/tsr_system.py`.
