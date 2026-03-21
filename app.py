import gradio as gr
import os
import time
import warnings
import logging
from src.pipeline import GameGen3DPipeline

# Ignorowanie ostrzeżeń Pandas, ResourceWarning i HuggingFace dla czystszego logu w terminalu
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

# Wyciszenie logerów Transformers / Diffusers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Inicjalizacja globalna potoku, by modele załadowały się do VRAM tylko raz na starcie
print("Inicjalizacja aplikacji WebUI...")
pipeline = GameGen3DPipeline()

def generate_3d_model(prompt: str, style: str, force_new: bool):
    """
    Funkcja łącząca Gradio z naszym rurociągiem AI.
    """
    if not prompt.strip():
        return None, "Wpisz jakiś prompt!"
        
    output_filename = f"model_{int(time.time())}.glb"
    
    try:
        # Uruchomienie generowania z wybranym stylem i parametrem wymuszania
        model_path, stats, sfx_path, vlm_feedback, task_dir = pipeline.run(prompt, output_filename, style=style, force_new=force_new)
        
        # Pobieranie wygenerowanego widoku referencyjnego z folderu zadania
        ref_path = os.path.join(task_dir, "views", "internet_reference.png") if task_dir else None
        preview_img = ref_path if ref_path and os.path.exists(ref_path) else None
        
        return model_path, preview_img, stats, sfx_path, vlm_feedback, "Sukces! Model 3D wygenerowany."
    except Exception as e:
        return None, None, {"error": str(e)}, None, {"error": str(e)}, f"Wystąpił błąd: {str(e)}"

def generate_scene(prompt: str, style: str):
    """
    Funkcja łącząca Gradio z rurociągiem do generowania całych scen.
    """
    if not prompt.strip():
        return None, "Wpisz jakiś prompt!"
        
    output_filename = f"scene_{int(time.time())}.glb"
    
    try:
        scene_path, layout = pipeline.run_scene(prompt, output_filename, style=style)
        return scene_path, layout, "Sukces! Scena wygenerowana."
    except Exception as e:
        return None, {"error": str(e)}, f"Wystąpił błąd: {str(e)}"

# Budowanie interfejsu Gradio
with gr.Blocks(title="GameGen-3D Studio") as app:
    gr.Markdown("# 🎲 GameGen-3D Studio")
    gr.Markdown("Kompleksowy system ML do generowania zasobów gamedev (Modele 3D, Sceny, Tekstury, SFX).")
    
    with gr.Tabs():
        with gr.Tab("Pojedynczy Obiekt (Single Object)"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Opis obiektu 3D (Prompt)", 
                        placeholder="np. stary drewniany fotel, brązowy",
                        lines=2
                    )
                    style_dropdown = gr.Dropdown(
                        choices=["Fotorealistyczny (PBR)", "Low Poly Art", "Voxel / Minecraft", "Cyberpunk / Sci-Fi", "Cel-Shaded (Anime)"],
                        value="Fotorealistyczny (PBR)",
                        label="Styl Wizualny (Art Style)",
                        info="Wybierz w jakim stylu ma zostać wygenerowany asset."
                    )
                    force_new_checkbox = gr.Checkbox(
                        label="Wymuś nową generację",
                        value=False,
                        info="Jeśli zaznaczone, system zignoruje pamięć wektorową (Cache) i wygeneruje model od zera."
                    )
                    generate_btn = gr.Button("🚀 Generuj Model 3D", variant="primary")
                    
                    status_text = gr.Textbox(label="Status", interactive=False)
                    
                    ref_image_output = gr.Image(
                        label="Znaleziona referencja (Rembg / DuckDuckGo)",
                        type="filepath",
                        interactive=False
                    )
                    
                    logic_output = gr.JSON(
                        label="Logika Gry (Wygenerowana przez LLM)",
                    )
                    
                    vlm_feedback_output = gr.JSON(
                        label="Ocena Art Directora (VLM - LLaVA)",
                    )
                    
                    sfx_output = gr.Audio(
                        label="Wygenerowany dźwięk (SFX)",
                        type="filepath",
                        interactive=False
                    )
                    
                with gr.Column():
                    model_3d_output = gr.Model3D(
                        label="Twój wygenerowany Model 3D (.glb)",
                        clear_color=[0.2, 0.2, 0.2, 1.0],
                        interactive=False
                    )

            # Podpięcie logiki
            generate_btn.click(
                fn=generate_3d_model,
                inputs=[prompt_input, style_dropdown, force_new_checkbox],
                outputs=[model_3d_output, ref_image_output, logic_output, sfx_output, vlm_feedback_output, status_text]
            )

        with gr.Tab("Generowanie Scen (Scene Generator)"):
            with gr.Row():
                with gr.Column():
                    scene_prompt_input = gr.Textbox(
                        label="Opis całej sceny (np. 'Pokój alchemika z drewnianym stołem i starym kotłem')", 
                        lines=3
                    )
                    scene_style_dropdown = gr.Dropdown(
                        choices=["Fotorealistyczny (PBR)", "Low Poly Art", "Voxel / Minecraft"],
                        value="Fotorealistyczny (PBR)",
                        label="Styl Wizualny"
                    )
                    scene_generate_btn = gr.Button("🌍 Generuj Scenę", variant="primary")
                    scene_status_text = gr.Textbox(label="Status", interactive=False)
                    scene_layout_output = gr.JSON(label="Zaplanowany układ przestrzenny (LLM Layout)")
                    
                with gr.Column():
                    scene_3d_output = gr.Model3D(
                        label="Wygenerowana Scena (.glb)",
                        clear_color=[0.2, 0.2, 0.2, 1.0],
                        interactive=False
                    )
                    
            scene_generate_btn.click(
                fn=generate_scene,
                inputs=[scene_prompt_input, scene_style_dropdown],
                outputs=[scene_3d_output, scene_layout_output, scene_status_text]
            )

if __name__ == "__main__":
    # Uruchomienie lokalnego serwera Gradio
    # share=False by zachować to na komputerze lokalnym (można zmienić na True, by udostępnić link)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Monochrome())
