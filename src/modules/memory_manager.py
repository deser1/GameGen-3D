import os
import json
import time
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self, db_path: str = "database/vector_cache"):
        """
        Inicjalizuje wektorową pamięć podręczną modeli 3D.
        Używa ChromaDB do przechowywania wektorów znaczeniowych i metadanych.
        """
        print("[Memory] Inicjalizacja wektorowej pamięci modeli 3D...")
        os.makedirs(db_path, exist_ok=True)
        os.makedirs("database/models_cache", exist_ok=True)
        
        # Inicjalizacja bazy ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="3d_models_cache")
        
        # Model do zamiany tekstu (promptu) na wektory liczbowe (Embeddings)
        # Używamy lekkiego i szybkiego modelu MiniLM
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _get_embedding(self, text: str) -> list[float]:
        """Zmienia prompt na wektor."""
        return self.embedding_model.encode(text).tolist()

    def check_memory(self, prompt: str, style: str, similarity_threshold: float = 0.88) -> str:
        """
        Przeszukuje bazę w poszukiwaniu modelu 3D o bardzo podobnym znaczeniu.
        Zwraca ścieżkę do pliku .glb, jeśli znajdzie "wspomnienie".
        """
        query_text = f"{prompt} in {style} style"
        query_embedding = self._get_embedding(query_text)
        
        # Przeszukiwanie bazy
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        if not results['distances'][0]:
            return None
            
        # W ChromaDB, w zależności od metryki (np. kosinusowa), mniejszy dystans = większe podobieństwo.
        # Domyślnie dystans < 0.25 oznacza bardzo duże podobieństwo (odpowiednik ~0.88 cosine similarity)
        distance = results['distances'][0][0]
        
        if distance < (1.0 - similarity_threshold):
            metadata = results['metadatas'][0][0]
            # Sprawdzamy czy styl też się zgadza
            if metadata.get('style') == style:
                cached_model_path = metadata.get('model_path')
                if os.path.exists(cached_model_path):
                    print(f"  [Memory] 🧠 Znaleziono wspomnienie! (Dystans: {distance:.3f})")
                    print(f"  [Memory] Używam zapisanego modelu z cache: {cached_model_path}")
                    return cached_model_path
        
        return None

    def save_to_memory(self, prompt: str, style: str, generated_lod0_path: str, seed: int = None):
        """
        Zapisuje wygenerowany model i jego parametry do pamięci trwałej,
        aby przyspieszyć przyszłe zapytania i umożliwić douczanie (Fine-Tuning).
        """
        timestamp = int(time.time())
        memory_id = f"mem_{timestamp}"
        
        # 1. Skopiowanie pliku 3D do bezpiecznego archiwum
        cached_model_path = f"database/models_cache/{memory_id}.glb"
        shutil.copy2(generated_lod0_path, cached_model_path)
        
        # 2. Zapisanie metadanych (do ewentualnego douczania sieci)
        metadata = {
            "prompt": prompt,
            "style": style,
            "timestamp": timestamp,
            "seed": seed if seed else "random",
            "model_path": cached_model_path
        }
        
        with open(f"database/models_cache/{memory_id}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        # 3. Zapis do bazy wektorowej
        query_text = f"{prompt} in {style} style"
        embedding = self._get_embedding(query_text)
        
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[memory_id]
        )
        print(f"  [Memory] 💾 Model pomyślnie zapamiętany na przyszłość (ID: {memory_id})")
