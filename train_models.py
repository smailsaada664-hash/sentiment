"""
========================================
Sentiment Analysis - FAST Training
========================================
ANN + LSTM only (quick version)

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import re

# Config
DATA_PATH = 'imdb.csv'
MAX_FEATURES = 10000
MAX_LEN = 200
BATCH_SIZE = 256
EPOCHS_ANN = 3
EPOCHS_LSTM = 2
VOCAB_SIZE = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.word_counts = {}
        
    def fit(self, texts):
        for text in texts:
            for word in text.split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:VOCAB_SIZE-2]):
            self.word2idx[word] = idx + 2
    
    def texts_to_sequences(self, texts):
        return [[self.word2idx.get(w, 1) for w in t.split()] for t in texts]
    
    def pad_sequences(self, sequences, maxlen=200):
        return [s[:maxlen] + [0]*(maxlen-len(s)) if len(s) < maxlen else s[:maxlen] for s in sequences]

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df['clean_text'] = df['sentences'].apply(preprocess_text)
print(f"Loaded {len(df)} samples")

# ============ ANN ============
print("\n" + "="*40)
print("TRAINING ANN")
print("="*40)

X = df['sentences'].values
y = df['labels'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

X_train_t = torch.FloatTensor(X_train_tfidf.toarray())
X_test_t = torch.FloatTensor(X_test_tfidf.toarray())
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)

class ANN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = ANN(MAX_FEATURES).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.BCEWithLogitsLoss()
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS_ANN):
    model.train()
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        loss = criterion(model(bx).squeeze(), by)
        loss.backward()
        opt.step()
    print(f"  ANN Epoch {epoch+1}/{EPOCHS_ANN}")

torch.save(model.state_dict(), 'ann_model.pth')
model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(X_test_t.to(device)).squeeze()) > 0.5).cpu().numpy()
ann_acc = accuracy_score(y_test, preds)
print(f"ANN Accuracy: {ann_acc:.4f}")

# ============ LSTM ============
print("\n" + "="*40)
print("TRAINING LSTM")
print("="*40)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    df['clean_text'].values, df['labels'].values, test_size=0.2, random_state=42
)

tok = SimpleTokenizer()
tok.fit(X_train_l)
X_train_seq = tok.texts_to_sequences(X_train_l)
X_test_seq = tok.texts_to_sequences(X_test_l)
X_train_pad = tok.pad_sequences(X_train_seq, MAX_LEN)
X_test_pad = tok.pad_sequences(X_test_seq, MAX_LEN)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tok, f)

X_train_lt = torch.LongTensor(X_train_pad)
X_test_lt = torch.LongTensor(X_test_pad)
y_train_lt = torch.FloatTensor(y_train_l)
y_test_lt = torch.FloatTensor(y_test_l)

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

lstm = LSTM().to(device)
opt = torch.optim.Adam(lstm.parameters(), lr=0.002)
loader = DataLoader(TensorDataset(X_train_lt, y_train_lt), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS_LSTM):
    lstm.train()
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        loss = criterion(lstm(bx).squeeze(), by)
        loss.backward()
        opt.step()
    print(f"  LSTM Epoch {epoch+1}/{EPOCHS_LSTM}")

torch.save(lstm.state_dict(), 'lstm_model.pth')
lstm.eval()
with torch.no_grad():
    preds = (torch.sigmoid(lstm(X_test_lt.to(device)).squeeze()) > 0.5).cpu().numpy()
lstm_acc = accuracy_score(y_test_l, preds)
print(f"LSTM Accuracy: {lstm_acc:.4f}")

# ============ BERT (placeholder) ============
print("\n" + "="*40)
print("SKIPPING BERT (requires transformers)")
print("="*40)
print("To train BERT, run: python train_bert.py")

print("\n" + "="*40)
print("DONE!")
print("="*40)
print(f"ANN Accuracy:  {ann_acc:.4f}")
print(f"LSTM Accuracy: {lstm_acc:.4f}")
print("\nStart API: python app.py")
