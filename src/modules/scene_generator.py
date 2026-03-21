import json
import ollama
import trimesh
import os

class SceneGeneratorLLM:
    def __init__(self, model_name: str = "llama3"):
        """
        Moduł generowania sceny (Spatial LLM Layouting).
        Używa LLM do rozbicia promptu opisującego całą scenę (np. 'pokój alchemika')
        na pojedyncze obiekty i ich koordynaty przestrzenne (x, y, z).
        """
        print("Inicjalizacja modułu Scene Generator (Spatial LLM)...")
        self.model_name = model_name

    def parse_scene_prompt(self, scene_prompt: str) -> list:
        """
        Pyta LLM o wygenerowanie układu (layoutu) sceny.
        Zwraca listę słowników: [{"item": "nazwa", "position": [x, y, z]}, ...]
        """
        system_prompt = (
            "You are an expert 3D level designer. The user will give you a description of a scene or room. "
            "Your task is to break it down into 3 to 5 individual 3D objects that make up this scene. "
            "For each object, provide its name (as a short prompt for a 3D generator) and its spatial position [x, y, z] in meters. "
            "The center of the room is [0, 0, 0]. Floor is y=0. "
            "Output strictly as a JSON array of objects. Example: "
            "[{\"item\": \"wooden table\", \"position\": [0, 0, 0]}, {\"item\": \"old chair\", \"position\": [1, 0, 0]}]"
        )
        
        try:
            print(f"  [SceneGen] Planowanie przestrzenne dla: '{scene_prompt}'...")
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': scene_prompt}
                ],
                format='json'
            )
            
            layout = json.loads(response['message']['content'])
            # W razie gdyby LLM ujął to w jakiś nadrzędny klucz
            if isinstance(layout, dict) and "objects" in layout:
                layout = layout["objects"]
            elif isinstance(layout, dict) and "items" in layout:
                layout = layout["items"]
                
            if not isinstance(layout, list):
                layout = [{"item": "mysterious object", "position": [0,0,0]}]
                
            print(f"  [SceneGen] LLM zaplanował {len(layout)} obiektów: {[i.get('item') for i in layout]}")
            return layout
            
        except Exception as e:
            print(f"  [SceneGen Błąd] Błąd podczas parsowania sceny: {e}")
            return [{"item": scene_prompt, "position": [0, 0, 0]}] # Fallback do 1 obiektu

    def merge_meshes(self, mesh_paths_with_positions: list, output_path: str) -> str:
        """
        Łączy wiele modeli 3D w jedną scenę (plik .glb) z zachowaniem przesunięć.
        mesh_paths_with_positions = [{"path": "model1.glb", "position": [0,0,0]}, ...]
        """
        print(f"  [SceneGen] Łączenie {len(mesh_paths_with_positions)} modeli w jedną scenę...")
        scene = trimesh.Scene()
        
        try:
            for item in mesh_paths_with_positions:
                path = item["path"]
                pos = item["position"]
                
                if not path or not os.path.exists(path):
                    continue
                    
                # Ładowanie pojedynczego mesha
                loaded = trimesh.load(path)
                
                # Upewnienie się, że to Mesh, a nie pod-scena (często .glb ładuje się jako Scene)
                if isinstance(loaded, trimesh.Scene):
                    if len(loaded.geometry) == 0:
                        continue
                    # Bierzemy pierwszy mesh z geometrii
                    geom_name = list(loaded.geometry.keys())[0]
                    mesh = loaded.geometry[geom_name]
                else:
                    mesh = loaded
                
                # Kopia, żeby nie modyfikować oryginału w cache
                mesh = mesh.copy()
                
                # Transformacja (Translacja)
                matrix = np.eye(4)
                matrix[0, 3] = float(pos[0])
                matrix[1, 3] = float(pos[1])
                matrix[2, 3] = float(pos[2])
                mesh.apply_transform(matrix)
                
                scene.add_geometry(mesh)
                
            # Zapisz połączoną scenę
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            scene.export(output_path)
            print(f"  [SceneGen] Sukces! Zapisano scenę do: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  [SceneGen Błąd] Błąd łączenia modeli: {e}")
            return None
