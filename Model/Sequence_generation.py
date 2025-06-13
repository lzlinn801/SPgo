import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import random
import numpy as np
import logging
import os
import subprocess

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO)

hydrophobicity = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

polar_amino_acids = ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W']

all_amino_acids = list(hydrophobicity.keys())

positive_amino_acids = ['K', 'R']
neutral_amino_acids = ['A', 'N', 'C', 'Q', 'G', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def load_fasta(file_path, tokenizer):
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
    return inputs['input_ids'], inputs['attention_mask']


def top_k_top_p_filtering(logits, top_k=20, top_p=0.9, filter_value=-10.0):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


class LSTMModel(nn.Module):
    def __init__(self, model_path, lstm_hidden_size=768, lstm_layers=3, dropout=0.3, bidirectional=True):
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

    def generate_sequence(self, input_ids, attention_mask, batch_size=256, top_k=50, top_p=0.9):
        self.eval()
        generated_sequences = []
        num_batches = (len(input_ids) + batch_size - 1) // batch_size

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(input_ids))

                batch_input_ids = input_ids[start_idx:end_idx]
                batch_attention_mask = attention_mask[start_idx:end_idx]

                model_outputs = self.protein_bert(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                last_hidden_state = model_outputs.last_hidden_state

                lstm_output, _ = self.lstm(last_hidden_state)

                for i in range(lstm_output.size(0)):
                    sequence = []
                    seq_length = lstm_output.size(1)

                    for t in range(seq_length):
                        output = self.fc(lstm_output[i, t, :])
                        logits = output.squeeze(0)
                        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                        probabilities = torch.softmax(filtered_logits, dim=-1)
                        token_id = torch.multinomial(probabilities, num_samples=1).item()
                        sequence.append(token_id)

                    generated_sequences.append(sequence)

        return generated_sequences


def c_sequence():
    while True:
        sequence = random.sample(polar_amino_acids, 3)
        X = random.choice(all_amino_acids)
        sequence += ['A', X, 'A']
        total_hydrophobicity = calculate_hydrophobicity(sequence)
        if total_hydrophobicity <= 1:
            return ''.join(sequence)


def n_sequence_v2(length=5):
    sequence = ['M']
    remaining = length - 1

    num_positive = np.random.randint(2, min(remaining, 4))  # 控制范围合理
    sequence += np.random.choice(positive_amino_acids, num_positive, replace=True).tolist()
    remaining -= num_positive

    if remaining > 0:
        sequence += np.random.choice(neutral_amino_acids, remaining, replace=True).tolist()

    random.shuffle(sequence[1:])
    return ''.join(sequence)


def calculate_hydrophobicity(sequence):
    return sum(hydrophobicity.get(aa, 0) for aa in sequence)


def run_signalp(input_file, output_file):
    try:
        command = [
            "signalp6",
            "--fasta", input_file,
            "--format", "txt",
            "--org", "other",
            "--output_dir", output_file
        ]

        subprocess.run(command, check=True)

        print(f"SignalP analysis completed successfully, results saved to {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error running SignalP: {e}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path ='bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_path)

    model = LSTMModel(model_path=model_path).to(device)
    model.load_state_dict(torch.load("checkpoint.pt", weights_only=True))

    input_ids, attention_mask = load_fasta('data/dataset.fasta', tokenizer)

    generated_sequence = model.generate_sequence(input_ids, attention_mask, top_k=20, top_p=0.9)

    decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True).upper() for seq in generated_sequence]

    valid_amino_acids = set(hydrophobicity.keys())
    filtered_sequences = []
    for seq in decoded_sequences:
        clean_seq = ''.join([aa for aa in seq if aa in valid_amino_acids])
        filtered_sequences.append(clean_seq)

    target_sequence = input("Target protein：").upper()
    output_file = 'result-seq.fasta'
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, seq in enumerate(filtered_sequences):
            csequences = c_sequence()
            nsequences = n_sequence()

            f.write(f">generated_signal_peptide_{i + 1}\n{nsequences}{seq}{csequences}{target_sequence}\n")

    logging.info(f"Generated sequences saved to {output_file}")
    input_file = "result-seq.fasta"
    output_file = "predict"
    run_signalp(input_file, output_file)
