from datasets import load_dataset
import json

print("Téléchargement du dataset financier (Format Parquet sécurisé)...")
# On utilise une version modernisée et certifiée Parquet par la communauté
dataset = load_dataset("lmassaron/FinancialPhraseBank", split="train")

output_file = "data/financial_dataset.jsonl"
sentiment_mapping = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}

print(f"Formatage de {len(dataset)} vraies phrases financières...")
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        # Récupération sécurisée du sentiment et conversion en français
        raw_sentiment = str(item.get("sentiment", "")).lower()
        final_sentiment = sentiment_mapping.get(raw_sentiment, "Neutre")
        
        row = {
            "instruction": "Analyse le sentiment de cette phrase financière et réponds uniquement par 'Positif', 'Négatif' ou 'Neutre'.",
            "input": item["sentence"],
            "output": final_sentiment
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Succès ! Fichier sauvegardé dans {output_file}")
