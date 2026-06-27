import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("⏳ Chargement du modèle de base et de l'adaptateur...")

# Nom du modèle de base et de ton modèle fine-tuné
base_model_name = "unsloth/llama-3-8b-bnb-4bit"
adapter_model_name = "FakhAhmed/slm-finance-sentiment"

# 1. Chargement du Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Chargement du modèle de base (en 4-bit pour économiser la RAM)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Fusion de l'adaptateur LoRA (Ton apprentissage) avec le modèle de base
model = PeftModel.from_pretrained(base_model, adapter_model_name)

def analyze_sentiment(text):
    prompt = f"Instruction: Tu es un expert financier. Analyse le sentiment de la phrase suivante (Positif, Négatif ou Neutre).\nTexte: {text}\nRéponse: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Génération
    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraction de la réponse finale
    sentiment = response.split("Réponse: ")[-1].strip()
    return sentiment

# Création de l'interface visuelle avec Gradio
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Entrez une phrase financière ici...", label="Texte Financier"),
    outputs=gr.Textbox(label="Sentiment Prédit (Positif / Négatif / Neutre)"),
    title="🏦 Financial SLM Sentiment Analyzer",
    description="Ce modèle est un Llama-3 8B fine-tuné avec LoRA pour l'analyse de sentiment financier. MLOps pipeline complet sur GCP.",
    examples=[
        ["Les revenus de l'entreprise ont chuté de 20% ce trimestre en raison de la crise."],
        ["Nous annonçons un dividende record et une croissance de 15% pour 2024."],
        ["La banque centrale a maintenu ses taux d'intérêt inchangés."]
    ]
)

if __name__ == "__main__":
    demo.launch()