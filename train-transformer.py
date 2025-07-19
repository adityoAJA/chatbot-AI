import json
import pandas as pd
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)

# === 1. LOAD JSON TO PANDAS ===
with open("samudra.json", "r") as f:
    intents = json.load(f)

data = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        data.append((pattern, tag))

df = pd.DataFrame(data, columns=["text", "label"])

# === 2. ENCODE LABEL ===
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])

# Simpan label mapping
label_map = {
    label: int(label_id)  # 👈 convert numpy.int32 to int
    for label, label_id in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
}
with open("label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# === 3. TRAIN/TEST SPLIT ===
train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)

# === 4. LOAD TOKENIZER & MODEL ===
model_name = "indobenchmark/indobert-base-p1"  # Bisa ganti ke 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenisasi
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

train_df = train_df[["text", "label_id"]]
train_dataset = Dataset.from_pandas(train_df.rename(columns={"label_id": "labels"})).map(tokenize, batched=True)

eval_df = eval_df[["text", "label_id"]]
eval_dataset = Dataset.from_pandas(eval_df.rename(columns={"label_id": "labels"})).map(tokenize, batched=True)

# === 5. LOAD MODEL ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Muat metrik akurasi dari library evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    # eval_pred adalah tuple yang berisi logits dan labels
    logits, labels = eval_pred
    
    # Ambil kelas dengan probabilitas tertinggi dari logits (argmax)
    predictions = np.argmax(logits, axis=-1)
    
    # Hitung akurasi
    return accuracy_metric.compute(predictions=predictions, references=labels)

# === 6. TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir="./bert-samudra-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    metric_for_best_model="accuracy",
    greater_is_better=True, # accuracy = True | loss = False
    save_total_limit=1
)

# === 7. TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# === 8. TRAIN ===
trainer.train()

print("✅ Model BERT berhasil dilatih dan disimpan!")
