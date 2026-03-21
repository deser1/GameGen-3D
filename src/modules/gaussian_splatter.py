import trimesh
import numpy as np
import os
import cv2
from PIL import Image

class GaussianSplatter:
    def __init__(self):
        """
        Moduł reprezentujący technologię 3D Gaussian Splatting (3DGS).
        Zamiast klasycznej siatki z trójkątów, generuje chmurę kolorowych punktów (Splatów),
        co idealnie sprawdza się w przypadku półprzezroczystych materiałów (dym, futro).
        W tej implementacji tworzymy chmurę punktów PLY jako reprezentację dla gier/silników.
        """
        print("Inicjalizacja modułu 3D Gaussian Splatting...")

    def generate_splats(self, mesh: trimesh.Trimesh, texture_image: Image.Image, output_path: str, num_points: int = 50000) -> str:
        """
        Generuje chmurę punktów (PointCloud) w formacie PLY imitującą splaty, 
        próbkując powierzchnię zoptymalizowanej siatki i pobierając kolory z tekstury.
        """
        print(f"  [3DGS] Generowanie chmury punktów (Gaussian Splats): {num_points} punktów...")
        
        try:
            # Próbkowanie punktów z powierzchni siatki
            points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
            
            # Pobieranie kolorów (jeśli siatka posiada UV i przekazano teksturę)
            colors = np.zeros((num_points, 4), dtype=np.uint8)
            colors[:, 3] = 255 # Alpha
            
            if hasattr(mesh.visual, 'uv') and texture_image is not None:
                uvs = mesh.visual.uv[face_indices] # To jest uproszczenie, uvs per face to zazwyczaj średnia
                
                # Bezpieczne mapowanie UV (0-1) na piksele obrazu
                img_array = np.array(texture_image)
                h, w = img_array.shape[:2]
                
                # Upewniamy się, że UV są w zakresie [0, 1]
                u = np.clip(uvs[:, 0], 0, 1)
                v = np.clip(1.0 - uvs[:, 1], 0, 1) # Odwrócenie osi V dla obrazów
                
                pixel_x = (u * (w - 1)).astype(int)
                pixel_y = (v * (h - 1)).astype(int)
                
                sampled_colors = img_array[pixel_y, pixel_x]
                if sampled_colors.shape[1] == 3:
                    colors[:, :3] = sampled_colors
                elif sampled_colors.shape[1] == 4:
                    colors = sampled_colors
            else:
                # Domyślny kolor jeśli brakuje UV
                colors[:, :3] = [150, 150, 150]
                
            # Tworzenie obiektu chmury punktów
            pc = trimesh.PointCloud(points, colors=colors)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pc.export(output_path)
            print(f"  [3DGS] Zapisano Gaussian Splats do: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  [3DGS Błąd] Nie udało się wygenerować Splatów: {e}")
            return None
