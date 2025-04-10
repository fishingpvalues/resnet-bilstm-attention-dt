from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


# --- Model Definition ---
class BiLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, attention_heads: int
    ):
        super(BiLSTM, self).__init__()

        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size if i == 0 else hidden_size * 2,
                    hidden_size,
                    batch_first=True,
                    dropout=0.3,
                    bidirectional=True,
                )
                for i in range(num_layers)
            ]
        )

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_size * 2, attention_heads)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.residual = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x)
            x = x.transpose(0, 1)
            x, _ = self.attention_layers[i](x, x, x)
            x = x.transpose(0, 1)
            x = self.layer_norm(x)
            residual = self.residual(x)
            x = F.relu(x + residual)
        x = torch.mean(x, dim=1)
        return torch.sigmoid(self.fc(x))


# --- Training and Evaluation Functions (unchanged) ---
def train_model(
    model: BiLSTM,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        verbose=True,
    )

    criterion = torch.nn.BCELoss()
    loss_history = []
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

    return loss_history


def evaluate_model(
    model: BiLSTM,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze(1)
            predicted = (outputs > 0.9).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_probs.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = correct / total if total > 0 else 0.0
    roc_auc = roc_auc_score(all_labels, all_probs)
    return {"accuracy": accuracy, "roc_auc": roc_auc}


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels_tensor = torch.stack(labels)
    return sequences_padded, labels_tensor


def diagnose_model(
    loss_history: List[float],
    accuracy: float,
    loss_threshold: float = 0.5,
    acc_threshold: float = 0.8,
) -> None:
    if loss_history and loss_history[-1] > loss_threshold:
        print("Warning: High loss, potential underfitting.")
    if accuracy < acc_threshold:
        print("Warning: Low accuracy, model may not capture process structure well.")
    if len(loss_history) >= 5 and (loss_history[0] - loss_history[-1]) < 0.1:
        print("Warning: Loss not decreasing significantly, possible underfitting.")


def evaluate_model_with_preds(
    model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze(1)
            probs = outputs.cpu().numpy().tolist()
            preds = (outputs > 0.9).float().cpu().numpy().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().tolist())
    return all_labels, all_preds, all_probs
