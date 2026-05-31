import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# ---------------------------------------------------------
# STEP 1: BILKUL FRESH DATA LOAD (with Balance & Shuffle)
# ---------------------------------------------------------
print("Datasets load ho rahe hain...")
df_fake = pd.read_csv('Fake.csv', low_memory=False).sample(n=1000, random_state=42)
df_real = pd.read_csv('True.csv', low_memory=False).sample(n=1000, random_state=42)
df_fake['label'] = 0  # Fake ab 0 hai
df_real['label'] = 1  # Real ab 1 hai

df = pd.concat([df_fake, df_real], ignore_index=True)

# Aisi cleaning jo sirf text par focus karegi
def deep_clean(text):
    text = str(text).lower()
    text = re.sub(r'reuters', '', text) # Reuters word hataya
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Sirf alphabets rakhien, baqi sab khatam
    return text.strip()

df['text'] = df['text'].apply(deep_clean)

# Shuffling taake model ko lagatar ek jesa data na miley
df = df.sample(frac=1, random_state=100).reset_index(drop=True)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.15, random_state=100
)

# ---------------------------------------------------------
# STEP 2: TOKENIZATION
# ---------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# ---------------------------------------------------------
# STEP 3: BERT TRAINING
# ---------------------------------------------------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16, # Batch size barha diya taake model bias na ho
    weight_decay=0.01, # Overfitting rokne ke liye
    eval_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=NewsDataset(train_encodings, train_labels),
    eval_dataset=NewsDataset(val_encodings, val_labels)
)

print("Training ho rahi hai... 2-3 minute sabr karein.")
trainer.train()

# ---------------------------------------------------------
# STEP 4: PREDICTION LOGIC WITH SAFEGUARD
# ---------------------------------------------------------
def predict_news(news_text):
    model.eval()
    news_text = deep_clean(news_text)
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Live outputs check karne ke liye logits print karein (Taake pata chalay model freeze toh nahi)
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    print(f"[Debug] Fake probability: {fake_prob:.2f} | Real probability: {real_prob:.2f}")

    if real_prob > fake_prob:
        return f"Real News ✅ ({real_prob*100:.1f}% confidence)"
    else:
        return f"Fake News 🚨 ({fake_prob*100:.1f}% confidence)"

print("\n" + "="*50)
print("MUKAMMAL WORKING MODEL READY!")
print("="*50)

while True:
    user_input = input("\nNews Likhein: ")
    if user_input.lower() == 'exit': break
    print(f"Result: {predict_news(user_input)}")