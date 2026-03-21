import os
import time
import sys

# Upewniamy się, że python znajdzie moduły
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.image_searcher import ImageReferenceSearcher
from src.modules.text_to_multiview import TextToMultiViewGenerator
from src.modules.multiview_to_3d import MultiViewTo3DReconstructor
from src.modules.mesh_optimizer import GameMeshOptimizer
from src.modules.memory_manager import MemoryManager
from src.modules.game_logic import GameLogicGenerator
from src.modules.art_director import ArtDirectorVLM
from src.modules.sfx_generator import SFXGenerator
from src.modules.gaussian_splatter import GaussianSplatter
from src.modules.texture_variants import TextureVariantGenerator
from src.modules.scene_generator import SceneGeneratorLLM
from src.modules.imagination import InternalKnowledgeGenerator
import json

class GameGen3DPipeline:
    def __init__(self):
        """
        Główna klasa orkiestrująca (Pipeline) łącząca moduły:
        1. Memory Manager
        2. Agent Wyszukujący
        3. Text to Multi-view (SD)
        4. Szybka Rekonstrukcja (TripoSR)
        5. Optimizer (Mesh, UV, PBR, LOD, Collision, Rig, Quad-Remesh, 4D Animation)
        6. Game Logic (LLM JSON)
        7. Art Director (VLM Feedback)
        8. SFX Generator (AudioLDM)
        9. Gaussian Splatting (PointCloud PLY)
        10. Texture Variants (Procedural Wear)
        11. Scene Generator (Spatial LLM)
        12. Imagination (Text-to-Image Fallback)
        """
        print("=== Inicjalizacja GameGen-3D Pipeline ===")
        
        # Inicjalizacja komponentów sztucznej inteligencji
        self.memory = MemoryManager()
        self.searcher = ImageReferenceSearcher()
        self.text_to_views = TextToMultiViewGenerator()
        self.views_to_3d = MultiViewTo3DReconstructor()
        self.mesh_optimizer = GameMeshOptimizer()
        self.logic_gen = GameLogicGenerator()
        self.art_director = ArtDirectorVLM()
        self.sfx_gen = SFXGenerator()
        self.gaussian = GaussianSplatter()
        self.texture_gen = TextureVariantGenerator()
        self.scene_gen = SceneGeneratorLLM()
        self.imagination = InternalKnowledgeGenerator()
        
        print("=== Gotowe do działania ===\n")

    def run(self, prompt: str, output_filename: str = "model.glb", style: str = "Fotorealistyczny (PBR)", force_new: bool = False, progress=None):
        """
        Uruchamia pełen proces od tekstu po model 3D gotowy dla silnika gry.
        Jeśli 'force_new' jest False, sprawdza czy model nie znajduje się już w pamięci podręcznej.
        """
        print(f">>> Start Pipeline dla zapytania: '{prompt}' (Styl: {style}) <<<")
        start_time = time.time()

        # Funkcja pomocnicza do raportowania postępu z czasem
        def report_progress(val, msg):
            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
            
            full_msg = f"[{time_str}] {msg}"
            if progress:
                progress(val, desc=full_msg)
            print(f"[{val*100:.0f}%] {full_msg}")

        # Etap -1: Pamięć Podręczna (Cache Check)
        report_progress(0.05, "Sprawdzanie pamięci wektorowej (Cache)...")
        stats = {}
        sfx_path = None
        vlm_feedback = None
        if not force_new:
            cached_model = self.memory.check_memory(prompt, style)
            if cached_model:
                report_progress(1.0, f"Znaleziono w pamięci! Odzyskiwanie modelu...")
                print(f">>> Model wyciągnięty z Pamięci Wektorowej w {time.time() - start_time:.2f} sekundy! <<<")
                
                # Tworzymy folder tasku nawet dla zcacheowanego modelu, żeby zachować spójność struktury
                import shutil
                from datetime import datetime
                import re
                
                safe_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt)[:20].strip('_').lower()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                task_folder_name = f"{timestamp}_{safe_prompt}"
                task_dir = os.path.join("output", task_folder_name)
                os.makedirs(task_dir, exist_ok=True)
                
                output_path = os.path.join(task_dir, output_filename)
                shutil.copy2(cached_model, output_path)
                
                # Zależnie od implementacji memory_manager, tu byśmy wyciągnęli też statystyki. 
                # Dla uproszczenia generujemy je na nowo dla cache'owanego modelu.
                stats = self.logic_gen.generate_stats(prompt, style)
                
                return output_path, stats, None, None, task_dir

        # --- Tworzenie dedykowanego folderu dla zadania (Task Folder) ---
        from datetime import datetime
        import re
        
        safe_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt)[:20].strip('_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_folder_name = f"{timestamp}_{safe_prompt}"
        task_dir = os.path.join("output", task_folder_name)
        
        # Zabezpieczenie katalogów wewnątrz folderu zadania
        views_dir = os.path.join(task_dir, "views")
        textures_dir = os.path.join(task_dir, "textures")
        sfx_dir = os.path.join(task_dir, "sfx")
        
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(views_dir, exist_ok=True)
        os.makedirs(textures_dir, exist_ok=True)
        os.makedirs(sfx_dir, exist_ok=True)
        
        print(f"  [Pipeline] Utworzono paczkę assetów (Task Folder): {task_dir}")

        # Modyfikacja promptu dla wyszukiwarki internetowej w zależności od stylu
        search_prompt = prompt
        if "Low Poly" in style:
            search_prompt += " low poly 3d model"
        elif "Voxel" in style:
            search_prompt += " voxel art minecraft style 3d"
        elif "Cyberpunk" in style:
            search_prompt += " cyberpunk sci-fi neon style"
        elif "Anime" in style:
            search_prompt += " anime cel shaded style"

        # Etap 0: Zebranie kontekstu wizualnego z Internetu
        report_progress(0.1, "Wyszukiwanie obrazu referencyjnego w sieci i wycinanie tła...")
        reference_img = self.searcher.search_reference(search_prompt)
        
        if reference_img:
            ref_path = os.path.join(views_dir, "internet_reference.png")
            reference_img.save(ref_path)
            print("  [Pipeline] Używanie znalezionego w sieci obrazu jako bazy.")
        else:
            report_progress(0.15, "Brak zdjęcia w sieci. Uruchamianie wewnętrznej wyobraźni AI (Text-to-Image)...")
            print("  [Pipeline] Fallback do wewnętrznego generatora wiedzy wizualnej.")
            reference_img = self.imagination.imagine_object(prompt, style)
            if reference_img:
                ref_path = os.path.join(views_dir, "imagined_reference.png")
                reference_img.save(ref_path)
            else:
                raise ValueError("Nie udało się ani znaleźć referencji, ani wygenerować jej z wyobraźni.")

        # Etap 1: Generowanie obrazów referencyjnych (Multi-view) na podstawie sieci z wstrzyknięciem stylu
        report_progress(0.25, "Generowanie 4 widoków (Multi-view) przy użyciu AI (Stable Diffusion)...")
        multi_views = self.text_to_views.generate(prompt, reference_image=reference_img, style=style)
        
        # Zapis podglądu wygenerowanych widoków
        for idx, img in enumerate(multi_views):
            img.save(os.path.join(views_dir, f"view_{idx}.png"))

        # Etap 2: Szybka Rekonstrukcja bryły (Raw 3D) na bazie map głębi / TripoSR
        report_progress(0.4, "Błyskawiczna rekonstrukcja modelu (TripoSR)...")
        raw_mesh = self.views_to_3d.reconstruct(multi_views)

        # Etap 3: Optymalizacja Game-Ready (Decimation & Smoothing)
        report_progress(0.5, "Czyszczenie geometrii i decymacja (Optymalizacja Gamedev)...")
        # TripoSR generuje dość czystą siatkę, decymacja zadba o Low-Poly
        low_poly_mesh = self.mesh_optimizer.optimize_mesh(raw_mesh, target_polycount=2500)

        # Etap 4: UV Unwrapping
        report_progress(0.55, "Rozkładanie siatki UV (Xatlas)...")
        self.mesh_optimizer.unwrap_uv(low_poly_mesh)

        # Etap 5: Teksturowanie (Albedo, Normal, Roughness) z użyciem Upscalera
        report_progress(0.6, "Wypalanie fotorealistycznych tekstur PBR...")
        # Optymalizator zapisuje tekstury lokalnie wokół mesha, dlatego musimy przekazać mu poprawny folder
        # (Wymaga małej zmiany w GameMeshOptimizer, jeśli pliki są zapisywane fizycznie - nasz korzysta z obiektów w locie i osadza je w glb)
        pbr_textures = self.mesh_optimizer.bake_pbr_textures(multi_views, low_poly_mesh, apply_upscale=True)

        # Etap 6: Zapis i Generowanie LODów (Export)
        report_progress(0.65, "Eksport modeli i generowanie poziomów detali (LOD)...")
        output_path = os.path.join(task_dir, output_filename)
        lod_files = self.mesh_optimizer.generate_lods(low_poly_mesh, output_path)

        # Etap 6.5: Generowanie siatki kolizyjnej do fizyki gry
        report_progress(0.7, "Generowanie siatek kolizyjnych dla silnika fizycznego...")
        collision_path = self.mesh_optimizer.generate_collision_mesh(low_poly_mesh, output_path)
        
        # Etap 6.6: Auto-Rigging (Tworzenie podstawowego szkieletu)
        report_progress(0.75, "Auto-Rigging (dodawanie szkieletu kości)...")
        # LLM na razie nie definiuje nam is_character przed logiką, więc spróbujemy to zgadnąć z promptu
        is_character = any(word in prompt.lower() for word in ["postać", "potwór", "ludzik", "zwierzę", "człowiek", "character", "monster"])
        rig_path = self.mesh_optimizer.auto_rig_model(low_poly_mesh, output_path, is_character=is_character)

        # Etap 6.7: 4D Animation (Blend Shapes / Morph Targets)
        report_progress(0.78, "Tworzenie animacji proceduralnych 4D (Morph Targets)...")
        anim_path = self.mesh_optimizer.generate_4d_animation(low_poly_mesh, output_path)

        # Etap 6.8: Gaussian Splatting (Generowanie PLY)
        report_progress(0.8, "Generowanie chmury punktów (Gaussian Splatting)...")
        base_name, _ = os.path.splitext(output_path)
        splat_path = self.gaussian.generate_splats(low_poly_mesh, pbr_textures["albedo"], f"{base_name}_splats.ply")

        # Etap 6.9: Warianty Tekstur (Procedural Wear)
        report_progress(0.85, "Proceduralne warianty tekstur (śnieg, rdza, zniszczenia)...")
        texture_variants = self.texture_gen.generate_variants(pbr_textures["albedo"], prompt, ["snowy", "mossy", "burnt"], output_dir=textures_dir)

        # Etap 7: Zapisanie wspomnienia (Zapis do bazy wektorowej ChromaDB)
        self.memory.save_to_memory(prompt, style, lod_files[0])
        
        # Etap 8: Generowanie statystyk JSON dla logiki gry
        report_progress(0.9, "Generowanie statystyk dla silnika gry (LLM)...")
        stats = self.logic_gen.generate_stats(prompt, style)
        
        # Zapiszmy statystyki do pliku JSON w folderze zadania
        if isinstance(stats, dict):
            import json
            stats_path = os.path.join(task_dir, "logic_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4)
                
            stats["extra_files"] = {
                "rig": rig_path,
                "animation_4d": anim_path,
                "gaussian_splats": splat_path,
                "texture_variants": texture_variants
            }

        # Etap 9: VLM Art Director Feedback (Ocena wizualna)
        report_progress(0.95, "Dyrektor Artystyczny ocenia wygenerowany model (VLM)...")
        vlm_feedback = self.art_director.review_model(prompt, multi_views[0])
        
        # Zapiszmy ocenę
        if vlm_feedback:
            import json
            feedback_path = os.path.join(task_dir, "art_director_feedback.json")
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(vlm_feedback, f, indent=4)
        
        # Etap 10: SFX Generation
        report_progress(0.98, "Generowanie efektów dźwiękowych (SFX)...")
        sfx_path = os.path.join(sfx_dir, f"sound_{int(time.time())}.wav")
        sfx_result = self.sfx_gen.generate_sfx(
            prompt=prompt, 
            output_path=sfx_path, 
            material=stats.get('material', 'unknown material') if isinstance(stats, dict) else 'unknown'
        )

        end_time = time.time()
        total_time = end_time - start_time
        m, s = divmod(int(total_time), 60)
        h, m = divmod(m, 60)
        final_time_str = f"{h:02d}:{m:02d}:{s:02d}"
        
        report_progress(1.0, f"Zakończono budowanie paczki! (Czas: {final_time_str})")
        print(f"\n>>> Zakończono z sukcesem w czasie {final_time_str} <<<")
        print(f"Paczka zasobów (Asset Pack) dostępna w: {task_dir}")
        
        return lod_files[0], stats, sfx_result, vlm_feedback, task_dir # Zwracamy task_dir dla interfejsu

    def run_scene(self, scene_prompt: str, output_filename: str = "scene.glb", style: str = "Fotorealistyczny (PBR)", progress=None):
        """
        Generuje całą scenę na podstawie promptu, używając Spatial LLM do zaplanowania obiektów.
        """
        print(f">>> Start Scene Pipeline dla zapytania: '{scene_prompt}' <<<")
        layout = self.scene_gen.parse_scene_prompt(scene_prompt)
        
        generated_items = []
        for i, item_data in enumerate(layout):
            item_name = item_data.get("item", "object")
            position = item_data.get("position", [0, 0, 0])
            print(f"\n--- Generowanie obiektu {i+1}/{len(layout)}: '{item_name}' w pozycji {position} ---")
            
            # Wymuszamy wygenerowanie pod-obiektu bez cache, by mieć unikalne modele (albo z cache, dla szybkości)
            # Używamy cache dla przyspieszenia generowania sceny
            try:
                model_path, _, _, _ = self.run(prompt=item_name, output_filename=f"scene_item_{i}.glb", style=style, force_new=False)
                generated_items.append({"path": model_path, "position": position})
            except Exception as e:
                print(f"Błąd podczas generowania '{item_name}': {e}")
                
        # Złączenie sceny
        scene_path = os.path.join("output", output_filename)
        final_path = self.scene_gen.merge_meshes(generated_items, scene_path)
        
        return final_path, layout

if __name__ == "__main__":
    # Testowy przebieg pipeline'u
    pipeline = GameGen3DPipeline()
    # Zmieniamy format na .glb (gltf binarny), który pakuje tekstury do jednego pliku
    pipeline.run("stara dębowa beczka wikingów z żelaznymi obręczami", "viking_barrel.glb")
