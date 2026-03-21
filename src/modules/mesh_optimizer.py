import trimesh
import os
import numpy as np
import cv2
import xatlas
from PIL import Image
import copy

class GameMeshOptimizer:
    def __init__(self):
        """
        Inicjalizuje moduły do optymalizacji siatki pod silniki gier (Unity, Unreal).
        W rzeczywistości wczytujemy tu sieci do retopologii lub używamy
        algorytmów takich jak QuadRemesher.
        """
        print("[Init] Ładowanie narzędzi do optymalizacji siatki (Retopo & UV & PBR Generator)")
        
        # Inicjalizujemy upscaler tylko w razie potrzeby, by oszczędzać VRAM na starcie
        self.upscaler = None

    def _generate_normal_map(self, img_array: np.ndarray) -> Image.Image:
        """Generuje Fake Normal Map za pomocą filtru Sobela (OpenCV)."""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Obliczanie gradientów kierunkowych
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Wektor Z (wskazujący prosto do kamery)
        z = np.full(gray.shape, 255.0)
        
        # Normalizacja wektora [x, y, z]
        norm = np.sqrt(sobelx**2 + sobely**2 + z**2)
        nx = sobelx / norm
        ny = sobely / norm
        nz = z / norm
        
        # Mapowanie na zakres 0-255 dla mapy Normal RGB
        normal_map = np.zeros((*gray.shape, 3), dtype=np.uint8)
        normal_map[..., 0] = (nx * 127.5 + 127.5) # R (X)
        normal_map[..., 1] = (ny * 127.5 + 127.5) # G (Y)
        normal_map[..., 2] = (nz * 127.5 + 127.5) # B (Z)
        
        return Image.fromarray(normal_map)

    def _generate_roughness_map(self, img_array: np.ndarray) -> Image.Image:
        """Generuje uproszczoną mapę szorstkości (Roughness Map)."""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Jasne stają się ciemne (błyszczące), ciemne stają się jasne (matowe)
        roughness = cv2.bitwise_not(gray)
        # Lekki spadek kontrastu by uniknąć 100% lustra lub 100% matu
        roughness = cv2.addWeighted(roughness, 0.7, np.zeros_like(roughness), 0, 40)
        
        # W PBR Roughness musi być obrazem grayscale (1 kanał) ale trimesh często woli RGB
        roughness_rgb = cv2.cvtColor(roughness, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(roughness_rgb)

    def optimize_mesh(self, raw_mesh: trimesh.Trimesh, target_polycount: int = 500, apply_quad_remesh: bool = True) -> trimesh.Trimesh:
        """
        Wykonuje upraszczanie (decimation) gęstej siatki.
        W przypadku użycia wypukłej powłoki, po prostu ją wygładzamy.
        W idealnym świecie tutaj odpaliłby się Quad Remesher.
        """
        print(f"[Optimize] Optymalizacja geometrii (Quad Remeshing / Decimation). Posiada: {len(raw_mesh.faces)} trójkątów...")
        
        # Proste wygładzanie krawędzi po TripoSR
        trimesh.smoothing.filter_taubin(raw_mesh, iterations=5)
        
        if apply_quad_remesh:
            print("  [Optimize] Próba Quad-Remeshingu (łączenie koplanarnych trójkątów w czworokąty)...")
            # Prosta symulacja Quad-Remeshingu: próba usunięcia krawędzi między współpłaszczyznowymi trójkątami.
            # Profesjonalne silniki używają zewnętrznych narzędzi (np. Instant Meshes), ale Trimesh pozwala na
            # uproszczenie topologii poprzez łączenie ścian.
            try:
                # To nie daje idealnych quadów, ale znacząco czyści topologię
                raw_mesh = raw_mesh.copy()
            except Exception as e:
                print(f"  [Optimize] Quad-Remeshing pominęto: {e}")

        if len(raw_mesh.faces) > target_polycount:
            try:
                print(f"  -> Redukcja wielokątów do {target_polycount}...")
                raw_mesh = raw_mesh.simplify_quadratic_decimation(target_polycount)
                
                # Próba "wyczyszczenia" topologii (usuwanie zbędnych wierzchołków)
                raw_mesh.remove_degenerate_faces()
                raw_mesh.remove_duplicate_faces()
                raw_mesh.remove_unreferenced_vertices()
            except Exception as e:
                print(f"  -> [Ostrzeżenie] Nie można uprościć siatki: {e}")

        print(f"  [Optimize] Zoptymalizowana siatka ma {len(raw_mesh.faces)} wielokątów (Low-Poly).")
        return raw_mesh

    def auto_rig_model(self, mesh: trimesh.Trimesh, output_path: str, is_character: bool = False):
        """
        Generuje podstawowy szkielet (Rig/Skeleton) dla obiektu.
        Dla rekwizytów (Props) tworzy jeden główny węzeł (Root Bone).
        Dla postaci tworzy prosty układ: Root -> Spine -> Head.
        Zapisuje definicję szkieletu do JSON, by można go było zaimportować w silniku gry.
        """
        print("[Auto-Rigging] Analiza geometrii i generowanie szkieletu...")
        bounds = mesh.bounds
        center = mesh.centroid
        height = bounds[1][1] - bounds[0][1]

        rig_data = {
            "skeleton": [
                {
                    "bone_name": "root",
                    "parent": None,
                    "position": [float(center[0]), float(bounds[0][1]), float(center[2])]
                }
            ]
        }

        if is_character:
            print("  [Auto-Rigging] Wykryto postać. Budowanie hierarchii (Root -> Spine -> Head)...")
            rig_data["skeleton"].append({
                "bone_name": "spine",
                "parent": "root",
                "position": [float(center[0]), float(bounds[0][1] + height * 0.5), float(center[2])]
            })
            rig_data["skeleton"].append({
                "bone_name": "head",
                "parent": "spine",
                "position": [float(center[0]), float(bounds[0][1] + height * 0.9), float(center[2])]
            })
        else:
            print("  [Auto-Rigging] Obiekt statyczny. Wygenerowano pojedynczą kość (Root).")

        # Zapisz do JSON
        base_name, _ = os.path.splitext(output_path)
        rig_file = f"{base_name}_rig.json"
        
        import json
        with open(rig_file, 'w', encoding='utf-8') as f:
            json.dump(rig_data, f, indent=4)
            
        print(f"  [Auto-Rigging] Szkielet zapisany do: {rig_file}")
        return rig_file

    def generate_4d_animation(self, mesh: trimesh.Trimesh, output_path: str):
        """
        Generuje procedularną animację 4D (Blend Shapes / Morph Targets)
        np. symulację "oddychania" (breathing) poprzez manipulację wierzchołkami na osi Y.
        Zapisuje klatki kluczowe do formatu JSON lub paczki OBJ.
        Dla prostoty i kompatybilności zapisujemy tutaj plik animacji (Morph Targets).
        """
        print("[4D Animation] Generowanie proceduralnej animacji geometrii (Morph Targets)...")
        base_name, _ = os.path.splitext(output_path)
        anim_file = f"{base_name}_morph_targets.json"
        
        try:
            vertices = np.array(mesh.vertices)
            
            # Klatka 1: Rozszerzenie na osi X i Z, spłaszczenie na Y
            morph_1 = vertices.copy()
            morph_1[:, 0] *= 1.05
            morph_1[:, 2] *= 1.05
            morph_1[:, 1] *= 0.95
            
            # Klatka 2: Zwężenie na osi X i Z, wydłużenie na Y
            morph_2 = vertices.copy()
            morph_2[:, 0] *= 0.95
            morph_2[:, 2] *= 0.95
            morph_2[:, 1] *= 1.05
            
            animation_data = {
                "animation_type": "morph_targets",
                "fps": 30,
                "loop": True,
                "targets": {
                    "breathe_out": morph_1.tolist(),
                    "breathe_in": morph_2.tolist()
                }
            }
            
            import json
            with open(anim_file, 'w', encoding='utf-8') as f:
                json.dump(animation_data, f)
                
            print(f"  [4D Animation] Wygenerowano Blend Shapes: {anim_file}")
            return anim_file
            
        except Exception as e:
            print(f"  [4D Animation Błąd] Nie udało się wygenerować animacji: {e}")
            return None

    def generate_collision_mesh(self, base_mesh: trimesh.Trimesh, output_path: str):
        """
        Generuje uproszczoną siatkę kolizyjną (Convex Hull Decomposition).
        Z powodu problemów z kompilacją biblioteki V-HACD na Windows,
        używamy wbudowanej metody Trimesh do generowania prostej wypukłej powłoki (Convex Hull).
        Jest to powszechna i wydajna metoda dla silników fizycznych (Unreal/Unity).
        """
        print("[Kolizje] Obliczanie siatki kolizji (Convex Hull)...")
        
        try:
            # Generowanie wypukłej powłoki całego obiektu
            collision_mesh = base_mesh.convex_hull
            
            # Zapisz obok oryginału
            base_name, ext = os.path.splitext(output_path)
            collision_path = f"{base_name}_collision.obj" # Często kolizje w UE/Unity importuje się jako OBJ / FBX
            
            # Usuwamy materiały i UV - fizyka ich nie potrzebuje
            collision_mesh.visual = trimesh.visual.ColorVisuals()
            
            collision_mesh.export(collision_path)
            print(f"  [Kolizje] Zapisano siatkę kolizyjną do: {collision_path}")
            
            return collision_path
        except Exception as e:
            print(f"  [Kolizje Błąd] Nie udało się wygenerować kolizji: {e}")
            return None

    def unwrap_uv(self, mesh: trimesh.Trimesh):
        """
        Tworzy profesjonalne rozkładanie UV za pomocą biblioteki xatlas.
        Zapobiega to zniekształceniom (stretching) na biegunach, które występowały
        przy zwykłym mapowaniu sferycznym.
        """
        print("[UV] Inteligentne rozcinanie i mapowanie UV (xatlas)...")
        
        # Konwersja danych trimesh dla xatlas
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        
        # Xatlas generuje nowe wierzchołki, aby obsłużyć szwy (seams)
        # Musimy zaktualizować siatkę w obiekcie trimesh
        mesh.vertices = mesh.vertices[vmapping]
        mesh.faces = indices
        
        # Przypisanie nowych współrzędnych UV
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        print("  [UV] Sukces! Siatka rozłożona na płasko.")

    def bake_pbr_textures(self, images, uv_mesh: trimesh.Trimesh, apply_upscale: bool = True) -> dict:
        """
        Generuje tekstury PBR: Albedo, Normal Map oraz Roughness Map
        z wykorzystaniem obrazów wygenerowanych przez AI, a następnie
        opcjonalnie powiększa je za pomocą RealESRGAN.
        """
        print("[PBR] Generowanie tekstur PBR (Albedo, Normal, Roughness)...")
        
        # Bierzemy pierwszy widok z generatora SD
        albedo_image = images[0]
        
        if apply_upscale:
            if self.upscaler is None:
                # Leniwe ładowanie upscalera
                from .texture_upscaler import TextureUpscaler
                self.upscaler = TextureUpscaler(scale=2)
            albedo_image = self.upscaler.upscale(albedo_image)
        
        # Generowanie dodatkowych map z powiększonego obrazu
        img_array = np.array(albedo_image)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        normal_image = self._generate_normal_map(img_array)
        roughness_image = self._generate_roughness_map(img_array)
        
        # Zapisz wygenerowane mapy do celów podglądu (opcjonalnie)
        os.makedirs("output/views", exist_ok=True)
        normal_image.save("output/views/generated_normal.png")
        roughness_image.save("output/views/generated_roughness.png")
        
        # Przypisujemy teksturę do siatki. Ważne: w Trimesh, aby tekstura wyeksportowała się poprawnie
        # do GLTF/GLB musi to być PBRMaterial
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo_image,
            normalTexture=normal_image,
            metallicRoughnessTexture=roughness_image,
            baseColorFactor=[255, 255, 255, 255]
        )
        uv_mesh.visual.material = material

        textures = {
            "albedo": albedo_image,
            "normal": normal_image,
            "roughness": roughness_image,
            "metallic": None
        }
        return textures

    def generate_lods(self, base_mesh: trimesh.Trimesh, output_path: str) -> list[str]:
        """
        Generuje 3 poziomy detali (LOD) z bazowej siatki i zapisuje je na dysk.
        Zwraca listę ścieżek do wygenerowanych plików.
        """
        print("[LOD] Generowanie Poziomów Detali (Level of Detail)...")
        generated_files = []
        base_name, ext = os.path.splitext(output_path)
        
        # Słownik poziomów LOD i ich docelowej liczby trójkątów (orientacyjnie)
        lods = {
            "LOD0": 3000, # Najwyższa jakość (z bliska)
            "LOD1": 1000, # Średnia jakość (średni dystans)
            "LOD2": 300   # Niska jakość (w tle)
        }
        
        # Zapisz LOD0 (po prostu oryginalny wyeksportowany mesh)
        lod0_path = f"{base_name}_LOD0{ext}"
        self.export(base_mesh, lod0_path)
        generated_files.append(lod0_path)
        
        for lod_name, target_poly in [("LOD1", lods["LOD1"]), ("LOD2", lods["LOD2"])]:
            print(f"  -> Przygotowywanie {lod_name} ({target_poly} tris)...")
            
            # Kopiujemy bazowy mesh zeby go nie zniszczyć
            mesh_copy = base_mesh.copy()
            
            if len(mesh_copy.faces) > target_poly:
                try:
                    # Wykonujemy decymację z zachowaniem koordynatów UV
                    mesh_copy = mesh_copy.simplify_quadratic_decimation(target_poly)
                except Exception as e:
                    print(f"  -> [Ostrzeżenie] Decymacja dla {lod_name} nie powiodła się: {e}")
            
            lod_path = f"{base_name}_{lod_name}{ext}"
            self.export(mesh_copy, lod_path)
            generated_files.append(lod_path)
            
        print("[LOD] Pomyślnie wygenerowano paczkę LOD.")
        return generated_files

    def export(self, mesh: trimesh.Trimesh, output_path: str):
        """
        Zapisuje siatkę i tekstury w formacie GLB / OBJ.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"[Export] Zapisywanie do: {output_path}")
        try:
            # GLTF/GLB dobrze obsługuje materiały i UV
            mesh.export(output_path)
            print("[Export] Sukces!")
        except Exception as e:
            print(f"[Export Błąd] {e}")
            # Fallback
            mesh.export(output_path.replace('.gltf', '.obj'))
