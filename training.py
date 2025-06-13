import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LSTMModel(nn.Module):
    def __init__(self, model_path, lstm_hidden_size=768, lstm_layers=6, dropout=0.3, bidirectional=True):
        super().__init__()
        self.protein_bert = BertModel.from_pretrained(model_path)
        
        for name, param in self.protein_bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=self.protein_bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.fc = nn.Linear(lstm_hidden_size * (2 if bidirectional else 1), self.protein_bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            model_outputs = self.protein_bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = model_outputs.last_hidden_state
        lstm_output, _ = self.lstm(last_hidden_state)
        output = self.fc(lstm_output)
        return output



def load_fasta_data(file_path, tokenizer):
    sequences = []
    with open(file_path, 'r') as f:
        sequence = ''
        for line in f:
            if line.startswith('>'):  
                if sequence:  
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line.strip()  
        if sequence:  
            sequences.append(sequence)

    
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    
    dataset = TensorDataset(input_ids, attention_mask)
    return dataset


def train_model(model, train_loader, val_loader, device, epochs, learning_rate, save_path, log_dir):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            optimizer.zero_grad()
            with autocast():  
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, model.protein_bert.config.vocab_size), input_ids.view(-1))

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        
        val_loss = validate(model, val_loader, device, criterion)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model to {save_path}")

        scheduler.step(val_loss)


def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, model.protein_bert.config.vocab_size), input_ids.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "bert-base-uncased"
    data_path = 'dataset.fasta'  
    save_path = "checkpoint.pt"
    batch_size = 256
    epochs = 20
    learning_rate = 1e-4
    tokenizer = BertTokenizer.from_pretrained(model_path)
    dataset = load_fasta_data(data_path, tokenizer)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LSTMModel(model_path=model_path)

    train_model(model, train_loader, val_loader, device, epochs, learning_rate, save_path, log_dir="./logs")
