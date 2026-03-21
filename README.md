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
# Instalacja wymaganych bibliotek (PyTorch, Diffusers, TripoSR, itp.)
pip install -r requirements.txt
```

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
