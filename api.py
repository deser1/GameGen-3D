from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import shutil
from src.pipeline import GameGen3DPipeline
from src.task_manager import task_manager

app = FastAPI(
    title="GameGen-3D API",
    description="API do generowania modeli 3D z tekstu dla silników gier.",
    version="1.0.0"
)

# CORS middleware to allow requests from game engines or web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serwowanie wygenerowanych plików (np. dla silnika Unity/UE)
os.makedirs("output", exist_ok=True)
app.mount("/files", StaticFiles(directory="output"), name="files")

# Serwowanie plików statycznych frontendu (Web UI)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Inicjalizacja GameGen-3D Pipeline dla API...")
pipeline = GameGen3DPipeline()

class GenerateRequest(BaseModel):
    prompt: str
    style: str = "Fotorealistyczny (PBR)"
    force_new: bool = False

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

def run_pipeline_task(task_id: str, prompt: str, style: str, force_new: bool):
    """Funkcja uruchamiana w tle (Background Task) realizująca pełne generowanie."""
    try:
        output_filename = f"model_{int(time.time())}.glb"
        
        # Funkcja callback (zastępująca gr.Progress) raportująca do TaskManager
        def progress_callback(val, desc=""):
            task_manager.update_task_progress(task_id, val, desc)
            
        model_path, stats, sfx_path, vlm_feedback, task_dir = pipeline.run(
            prompt=prompt, 
            output_filename=output_filename, 
            style=style, 
            force_new=force_new,
            progress=progress_callback
        )
        
        # Budowanie adresów URL
        model_url = "/" + model_path.replace("\\", "/").replace("output/", "files/", 1)
        
        ref_path_internet = os.path.join(task_dir, "views", "internet_reference.png") if task_dir else None
        ref_path_imagined = os.path.join(task_dir, "views", "imagined_reference.png") if task_dir else None
        
        ref_url = None
        if ref_path_internet and os.path.exists(ref_path_internet):
            ref_url = "/" + ref_path_internet.replace("\\", "/").replace("output/", "files/", 1)
        elif ref_path_imagined and os.path.exists(ref_path_imagined):
            ref_url = "/" + ref_path_imagined.replace("\\", "/").replace("output/", "files/", 1)
            
        sfx_url = "/" + sfx_path.replace("\\", "/").replace("output/", "files/", 1) if sfx_path and os.path.exists(sfx_path) else None
        
        result_data = {
            "model_url": model_url,
            "reference_url": ref_url,
            "stats": stats,
            "sfx_url": sfx_url,
            "vlm_feedback": vlm_feedback,
            "task_dir": task_dir.replace("\\", "/") # Zwracamy katalog by móc go pobrać jako zip
        }
        
        task_manager.mark_task_completed(task_id, result_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.mark_task_failed(task_id, str(e))


@app.post("/api/generate", response_model=TaskResponse)
async def generate_model_async(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Endpoint zlecający generowanie modelu. Zwraca od razu Task ID,
    który można odpytywać, aby uniknąć problemu zrywania połączeń w przeglądarce.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt nie może być pusty.")

    # Tworzenie nowego zadania
    task_id = task_manager.create_task(req.prompt, req.style)
    
    # Dodanie zadania do wykonania w tle przez serwer FastAPI
    background_tasks.add_task(run_pipeline_task, task_id, req.prompt, req.style, req.force_new)
    
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Zadanie zostało zlecone do kolejki."
    )

class GenerateSceneRequest(BaseModel):
    prompt: str
    style: str = "Fotorealistyczny (PBR)"

def run_scene_task(task_id: str, prompt: str, style: str):
    """Funkcja uruchamiana w tle realizująca generowanie całej sceny."""
    try:
        output_filename = f"scene_{int(time.time())}.glb"
        
        def progress_callback(val, desc=""):
            task_manager.update_task_progress(task_id, val, desc)
            
        scene_path, layout = pipeline.run_scene(
            scene_prompt=prompt, 
            output_filename=output_filename, 
            style=style, 
            progress=progress_callback
        )
        
        scene_url = "/" + scene_path.replace("\\", "/").replace("output/", "files/", 1)
        
        result_data = {
            "model_url": scene_url,
            "stats": {"layout": layout},
            "reference_url": None,
            "sfx_url": None,
            "vlm_feedback": None
        }
        
        task_manager.mark_task_completed(task_id, result_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.mark_task_failed(task_id, str(e))

@app.post("/api/generate_scene", response_model=TaskResponse)
async def generate_scene_async(req: GenerateSceneRequest, background_tasks: BackgroundTasks):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt sceny nie może być pusty.")

    task_id = task_manager.create_task(req.prompt, req.style)
    background_tasks.add_task(run_scene_task, task_id, req.prompt, req.style)
    
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Zadanie generowania sceny zostało zlecone do kolejki."
    )

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Endpoint zwracający aktualny status generowania (do paska postępu w UI)."""
    return task_manager.get_task_status(task_id)

@app.get("/api/download/{task_folder}")
async def download_task_zip(task_folder: str):
    """Pakuje cały folder zadania do formatu ZIP i zwraca go do pobrania."""
    folder_path = os.path.join("output", task_folder)
    
    # Ochrona przed path traversal
    if not os.path.exists(folder_path) or ".." in task_folder or "/" in task_folder or "\\" in task_folder:
        raise HTTPException(status_code=404, detail="Nie znaleziono folderu zadania.")
        
    zip_path = os.path.join("output", f"{task_folder}.zip")
    
    # Tworzymy archiwum zip jeśli jeszcze nie istnieje
    if not os.path.exists(zip_path):
        shutil.make_archive(folder_path, 'zip', folder_path)
        
    return FileResponse(zip_path, media_type="application/zip", filename=f"{task_folder}_GameGen3D.zip")

@app.get("/")
async def serve_frontend():
    """Główny interfejs webowy odporny na odświeżanie."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API serwera GameGen-3D działa poprawnie."}

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Konfiguracja niestandardowego filtra logów, aby usunąć spam związany z odpytywaniem o status zadania co sekundę
    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            # Filtrujemy tylko wiadomości, które zawierają konkretne ścieżki (status i health)
            msg = record.getMessage()
            if msg.find("GET /api/status/") != -1 or msg.find("GET /health") != -1:
                # Jeśli to jest zapytanie zakończone błędem (np. 404 lub 500), chcemy je zobaczyć
                if record.levelno >= logging.WARNING or " 200 " not in msg:
                    return True
                # W przeciwnym razie wyciszamy
                return False
            return True

    # Dodajemy filtr do loggera dostępu Uvicorna
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

    # Uruchomienie lokalnego serwera API
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
