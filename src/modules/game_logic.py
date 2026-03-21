import json
import ollama

class GameLogicGenerator:
    def __init__(self, model_name: str = "llama3"):
        """
        Inicjalizuje moduł generowania logiki gry za pomocą lokalnego LLM (Ollama).
        Pozwala na wygenerowanie sensownych statystyk i metadanych dla wygenerowanego modelu 3D.
        """
        self.model_name = model_name
        print(f"[Init] Ładowanie generatora logiki gry (LLM: {self.model_name})...")
        
        # Sprawdzenie czy ollama jest dostępna
        try:
            ollama.list()
            self.available = True
        except Exception:
            print("[Ostrzeżenie] Nie wykryto działającego serwera Ollama. Generator logiki użyje domyślnych wartości (Fallback).")
            self.available = False

    def generate_stats(self, prompt: str, style: str) -> dict:
        """
        Generuje plik JSON ze statystykami przedmiotu na podstawie jego opisu.
        """
        print(f"[GameLogic] Generowanie metadanych dla: {prompt}")
        
        if not self.available:
            # Fallback jeśli nie ma LLMa
            return {
                "item_name": prompt,
                "type": "Prop",
                "weight_kg": 10.0,
                "is_destructible": False,
                "material": "Unknown",
                "rarity": "Common",
                "description": f"A {style} generated object."
            }

        # System prompt wymuszający format JSON
        system_prompt = """You are a senior game designer. 
Your task is to generate balanced, realistic game item statistics based on a user's description.
You MUST respond ONLY with a valid JSON object, no markdown formatting, no explanations.
The JSON must contain these exact keys:
- "item_name" (string, cool name for the item)
- "type" (string, e.g. Weapon, Armor, Consumable, Prop, Container)
- "weight_kg" (float, realistic weight)
- "is_destructible" (boolean)
- "material" (string, primary material e.g. Wood, Metal, Stone)
- "rarity" (string: Common, Uncommon, Rare, Epic, Legendary)
- "value_gold" (integer, base price)
- "description" (string, a short lore-friendly flavor text)"""

        user_prompt = f"Generate stats for this item: '{prompt}' in '{style}' style."

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ], format='json')
            
            # Parsowanie odpowiedzi do obiektu Python
            stats = json.loads(response['message']['content'])
            print(f"  [GameLogic] Sukces! Wygenerowano: {stats.get('item_name')}")
            return stats
            
        except Exception as e:
            print(f"  [GameLogic Błąd] Wystąpił błąd podczas generowania przez LLM: {e}")
            return {"error": "Failed to generate stats", "raw_prompt": prompt}
