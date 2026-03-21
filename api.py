from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
from src.pipeline import GameGen3DPipeline

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

print("Inicjalizacja GameGen-3D Pipeline dla API...")
pipeline = GameGen3DPipeline()

class GenerateRequest(BaseModel):
    prompt: str
    style: str = "Fotorealistyczny (PBR)"
    force_new: bool = False

class GenerateResponse(BaseModel):
    status: str
    model_url: str
    reference_url: str | None
    stats: dict
    sfx_url: str | None
    vlm_feedback: dict | None

@app.post("/generate", response_model=GenerateResponse)
async def generate_model(req: GenerateRequest):
    """
    Endpoint generujący model 3D na podstawie promptu.
    Uwaga: W środowisku produkcyjnym generowanie synchroniczne potrwa chwilę.
    Dla długich zadań można użyć Celery lub BackgroundTasks z poolingiem.
    Tutaj dla prostoty trzymamy połączenie.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt nie może być pusty.")

    output_filename = f"model_{int(time.time())}.glb"
    
    try:
        model_path, stats, sfx_path, vlm_feedback, task_dir = pipeline.run(
            prompt=req.prompt, 
            output_filename=output_filename, 
            style=req.style, 
            force_new=req.force_new
        )
        
        # Przygotowanie URL do wygenerowanych zasobów
        # Oczekujemy ścieżek względem głównego katalogu (np. output/task_name/model.glb)
        # i zamieniamy "output/" na nasz punkt montowania "/files/"
        model_url = "/" + model_path.replace("\\", "/").replace("output/", "files/", 1)
        
        ref_path = os.path.join(task_dir, "views", "internet_reference.png") if task_dir else None
        ref_url = "/" + ref_path.replace("\\", "/").replace("output/", "files/", 1) if ref_path and os.path.exists(ref_path) else None
        
        sfx_url = "/" + sfx_path.replace("\\", "/").replace("output/", "files/", 1) if sfx_path and os.path.exists(sfx_path) else None
        
        return GenerateResponse(
            status="success",
            model_url=model_url,
            reference_url=ref_url,
            stats=stats,
            sfx_url=sfx_url,
            vlm_feedback=vlm_feedback
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API serwera GameGen-3D działa poprawnie."}

if __name__ == "__main__":
    import uvicorn
    # Uruchomienie lokalnego serwera API
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
