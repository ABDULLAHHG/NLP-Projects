import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample text data (replace with your own dataset)
text = """This is a sample text for demonstrating a character-level autoregressive model.
It's relatively short, but sufficient for illustrating the basic principles."""

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
embedding_dim = 10
hidden_dim = 100
seq_length = 20
learning_rate = 0.01
num_epochs = 100

# Custom Dataset
class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]
        input_tensor = torch.tensor([char_to_ix[char] for char in input_seq])
        target_tensor = torch.tensor([char_to_ix[char] for char in target_seq])
        return input_tensor, target_tensor


# Model definition
class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AutoregressiveModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out) # Output shape is now (batch_size, seq_length, vocab_size)
        return out

# Create dataset and dataloader (unchanged)
dataset = CharDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True) #batch size is 1


# Initialize model, optimizer, and loss function (unchanged)
model = AutoregressiveModel(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")


# Generate text
def generate_text(model, start_char, length):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([char_to_ix[start_char]])
        generated_text = start_char
        for i in range(length):
            output = model(input_tensor.unsqueeze(0)) 
            predicted_index = torch.argmax(output[:, -1, :]).item()
            predicted_char = ix_to_char[predicted_index]
            generated_text += predicted_char
            input_tensor = torch.tensor([predicted_index])
    return generated_text

generated_text = generate_text(model, 'T', 50)
print(f"Generated text: {generated_text}")