import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Load dataset
# --------------------------

df = pd.read_parquet("train-00000-of-00001.parquet")

text = " ".join(df.astype(str).values.flatten())
text = text[:20000]

print("Text length:", len(text))

# --------------------------
# Character mapping
# --------------------------

chars = sorted(set(text))

char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

vocab_size = len(chars)

print("Vocabulary size:", vocab_size)

# encode text
encoded = np.array([char_to_idx[c] for c in text])

seq_length = 100

X = []
y = []

for i in range(len(encoded)-seq_length):
    X.append(encoded[i:i+seq_length])
    y.append(encoded[i+1:i+seq_length+1])

X = torch.tensor(X)
y = torch.tensor(y)

# --------------------------
# RNN Model
# --------------------------

class CharRNN(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)

        self.lstm = nn.LSTM(128, 256, batch_first=True)

        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        out,_ = self.lstm(x)

        out = self.fc(out)

        return out


model = CharRNN(vocab_size)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# --------------------------
# Training
# --------------------------

epochs = 5

losses = []

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X)

    loss = criterion(output.view(-1,vocab_size), y.view(-1))

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch+1} Loss:", loss.item())

# --------------------------
# Plot training
# --------------------------

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# --------------------------
# Generate text
# --------------------------

def generate_text(start="Hello", length=200):

    model.eval()

    input_seq = torch.tensor([[char_to_idx[c] for c in start]])

    result = start

    for i in range(length):

        output = model(input_seq)

        probs = torch.softmax(output[0,-1], dim=0)

        char_index = torch.multinomial(probs,1).item()

        result += idx_to_char[char_index]

        input_seq = torch.cat([input_seq[:,1:], torch.tensor([[char_index]])], dim=1)

    return result


print("\nGenerated Text:\n")

print(generate_text("Hello"))