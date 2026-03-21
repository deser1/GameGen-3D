import uuid
import time

class TaskManager:
    def __init__(self):
        # Słownik trzymający w pamięci RAM informacje o zadaniach
        # W rozwiązaniu produkcyjnym użylibyśmy Redis lub bazy danych SQLite
        self.tasks = {}

    def create_task(self, prompt: str, style: str) -> str:
        """Tworzy nowe zadanie i zwraca jego unikalne ID."""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Zadanie umieszczone w kolejce...",
            "prompt": prompt,
            "style": style,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        return task_id

    def update_task_progress(self, task_id: str, progress: float, message: str):
        """Aktualizuje postęp zadania. Może być używane jako callback w pipeline.py."""
        if task_id in self.tasks:
            self.tasks[task_id]["progress"] = progress
            self.tasks[task_id]["message"] = message
            self.tasks[task_id]["status"] = "processing"
            self.tasks[task_id]["updated_at"] = time.time()

    def mark_task_completed(self, task_id: str, result: dict):
        """Zaznacza zadanie jako zakończone sukcesem i zapisuje wyniki."""
        if task_id in self.tasks:
            self.tasks[task_id]["progress"] = 1.0
            self.tasks[task_id]["message"] = "Generowanie zakończone sukcesem!"
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["updated_at"] = time.time()

    def mark_task_failed(self, task_id: str, error_msg: str):
        """Zaznacza zadanie jako zakończone błędem."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = error_msg
            self.tasks[task_id]["message"] = f"Wystąpił błąd: {error_msg}"
            self.tasks[task_id]["updated_at"] = time.time()

    def get_task_status(self, task_id: str) -> dict:
        """Pobiera aktualny status zadania."""
        if task_id not in self.tasks:
            return {"status": "not_found", "message": "Nie znaleziono takiego zadania."}
        return self.tasks[task_id]

# Globalna instancja Managera Zadań
task_manager = TaskManager()
