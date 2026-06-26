cat << 'EOF' > src/train.py
import os
import mlflow
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

print("🚀 Démarrage du pipeline de Fine-Tuning SLM...")

# 1. Connexion à notre Tour de Contrôle MLflow (GCP Cloud Run)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("finance-sentiment-slm")
    print(f"✅ MLflow connecté : {tracking_uri}")
else:
    print("⚠️ MLFLOW_TRACKING_URI non défini. Tracking local.")

# 2. Chargement du Modèle avec Unsloth (Llama-3 8B ou un modèle plus petit)
max_seq_length = 2048
model_name = "unsloth/llama-3-8b-bnb-4bit" # Modèle quantifié pour Colab

print(f"📦 Chargement du modèle {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Configuration LoRA (Low-Rank Adaptation)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. Préparation du Dataset Financier (récupéré via DVC)
print("📊 Chargement des données financières...")
dataset = load_dataset("json", data_files="data/financial_dataset.jsonl", split="train")

def format_prompt(examples):
    prompts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = f"Instruction: {instruction}\nTexte: {input_text}\nRéponse: {output}"
        prompts.append(text)
    return {"text": prompts}

dataset = dataset.map(format_prompt, batched=True)

# 4. Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="models/slm-finance-v1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60, # Court pour le PoC
    learning_rate=2e-4,
    fp16=not getattr(model, "is_bf16_supported", lambda: False)(),
    bf16=getattr(model, "is_bf16_supported", lambda: False)(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="mlflow", # 🔴 LA MAGIE EST ICI : Dit à HuggingFace d'utiliser MLflow
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# 5. Lancement
with mlflow.start_run():
    print("🔥 Début de l'entraînement...")
    trainer_stats = trainer.train()
    print("✅ Entraînement terminé !")
    
    # Sauvegarde du modèle final
    model.save_pretrained("models/slm-finance-lora")
    tokenizer.save_pretrained("models/slm-finance-lora")
    print("💾 Modèle sauvegardé localement dans models/slm-finance-lora")
EOF