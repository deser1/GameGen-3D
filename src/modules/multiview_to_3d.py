from PIL import Image
import trimesh
import numpy as np
import torch
import sys
import os

# Dodajemy repozytorium TripoSR do ścieżki Pythona, aby można było z niego importować `tsr`
triposr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TripoSR_repo")
if os.path.exists(triposr_path) and triposr_path not in sys.path:
    sys.path.insert(0, triposr_path)

try:
    from tsr.system import TSR
    HAS_TSR = True
except ImportError as e:
    print(f"[Ostrzeżenie] Nie można zaimportować 'tsr' (TripoSR): {e}. Zastosowano fallback do trimesh.")
    HAS_TSR = False

class MultiViewTo3DReconstructor:
    def __init__(self):
        """
        Inicjalizuje model TripoSR (TSR) do błyskawicznej rekonstrukcji 3D z pojedynczego obrazu.
        Jest to model LRM (Large Reconstruction Model) bazujący na sieciach Transformer.
        """
        print(f"[Init] Ładowanie zaawansowanego modelu rekonstrukcji TripoSR...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        if HAS_TSR:
            try:
                # TripoSR automatycznie pobierze swoje wagi z HuggingFace Hub
                self.model = TSR.from_pretrained(
                    "stabilityai/TripoSR",
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                self.model.renderer.set_chunk_size(131072)
                self.model.to(self.device)
            except Exception as e:
                print(f"[Błąd] Nie udało się załadować TripoSR. Błąd: {e}")
                self.model = None

    def reconstruct(self, images: list[Image.Image], progress_callback=None) -> trimesh.Trimesh:
        """
        Przetwarza przedni widok wygenerowany przez SD (z usuniętym tłem)
        bezpośrednio w siatkę (mesh) za pomocą TripoSR.
        """
        print(f"[Reconstruct] Uruchamianie TripoSR LRM na głównym obrazie...")
        if progress_callback:
            progress_callback(0.41, "Uruchamianie sieci LRM (przewidywanie pola NeRF)...")
        
        if self.model is None:
            print("[Reconstruct] Fallback do icosphere (brak modelu)")
            return trimesh.creation.icosphere(subdivisions=3, radius=1.0)
            
        # TripoSR wymaga pojedynczego obrazka z usuniętym tłem (RGBA lub czyste tło)
        # Bierzemy Front View (indeks 0) wygenerowany przez SD
        input_image = images[0]
        
        # Upewniamy się, że obraz jest w formacie odpowiednim dla modelu
        if input_image.mode != "RGBA":
            # Jeśli tło jest czysto białe, TripoSR i tak sobie poradzi
            input_image = input_image.convert("RGB")
            
        # TripoSR przyjmuje listę obrazów dla batchowania, u nas to 1 obraz
        with torch.no_grad():
            scene_codes = self.model([input_image], device=self.device)
            
        # Ekstrakcja siatki metodą Marching Cubes z wbudowanego algorytmu TripoSR
        print("  -> Ekstrakcja siatki w wysokiej rozdzielczości...")
        # Zmniejszamy rozdzielczość Marching Cubes do 128 (lub 192), by zapobiec zamrażaniu procesu (CPU bottleneck z mcubes)
        
        # Przekazujemy callback do naszego lokalnego modelu TSR
        if hasattr(self.model, 'extract_mesh') and 'progress_callback' in self.model.extract_mesh.__code__.co_varnames:
            meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=192, progress_callback=progress_callback)
        else:
            # Fallback jeśli TSR nie wspiera callbacku
            meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=192)
        
        # Odbieramy pierwszy z wygenerowanych meshów
        tsr_mesh = meshes[0]
        
        # Konwersja formatu wyjściowego TripoSR na trimesh.Trimesh
        # TripoSR zwraca krotkę (vertices, faces, vertex_colors) lub obiekt zależny od implementacji
        # Aktualna wersja zwraca obiekt trimesh.Trimesh bezpośrednio
        
        if not isinstance(tsr_mesh, trimesh.Trimesh):
            # Fallback ręcznego pakowania jeśli nowsza wersja zmieni format
            raw_mesh = trimesh.Trimesh(
                vertices=tsr_mesh.vertices,
                faces=tsr_mesh.faces,
                vertex_colors=tsr_mesh.visual.vertex_colors if hasattr(tsr_mesh, 'visual') else None
            )
        else:
            raw_mesh = tsr_mesh
            
        print(f"[Reconstruct] Sukces! Wygenerowano siatkę 3D: {len(raw_mesh.faces)} trójkątów.")
        
        # TripoSR często generuje obiekty obrócone o 90 stopni wzgledem osi X w niektórych silnikach
        rot_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        raw_mesh.apply_transform(rot_matrix)
        
        return raw_mesh
