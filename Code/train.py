import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from net_lstm import LSTM_seq

def train(train_outputs, train_labels, vocab_size, device, epochs, batch_size, hidden_size, learning_rate):
    train_dataset = TensorDataset(train_outputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    lstm_model = LSTM_seq(
        max_seq=train_labels.size(1),
        input_size=train_outputs.size(-1),
        hidden_size=hidden_size,
        class_num=vocab_size,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    print("Starte Training...")
    for epoch in range(epochs):
        lstm_model.train()
        epoch_loss = 0
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            with autocast():
                output = lstm_model(batch_features, batch_labels)
                loss = criterion(output.reshape(-1, output.size(-1)), batch_labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training abgeschlossen. Speichere Modell...")

    # Unterordner "finished" erstellen, falls nicht vorhanden
    finished_dir = "finished"
    os.makedirs(finished_dir, exist_ok=True)

    # Modell speichern
    torch.save(lstm_model.state_dict(), os.path.join(finished_dir, "trained_lstm_model.pth"))
    print("Modell gespeichert.")

def save_test_data(test_outputs, test_labels):
    print("Speichere Testdaten...")
    finished_dir = "finished"
    os.makedirs(finished_dir, exist_ok=True)

    # Verpackte Testdaten speichern
    torch.save({
        "outputs": test_outputs,
        "labels": test_labels
    }, os.path.join(finished_dir, "processed_test_data.pt"))

    print("Testdaten gespeichert und bereit zur Validierung.")
