import argparse
from utils_packages import install_and_import

# Argumente mit Standardwerten definieren
parser = argparse.ArgumentParser(description="Datenvorverarbeitung und Modussteuerung")
parser.add_argument("--data_preprocessing_mode", type=str, choices=["process", "load"], default="load",
                    help="'process' oder 'load' für Datenvorverarbeitung (Standard: 'load')")
parser.add_argument("--step", type=int, choices=range(0, 8), default=0,
                    help="Vorverarbeitungsschritt (0 - 7, Standard: 0)")
parser.add_argument("--dataset_mode", type=int, choices=[0, 1, 2], default=0,
                    help="Dataset-Modus: 0 = Train und Test, 1 = Train, 2 = Test (Standard: 0)")
args = parser.parse_args()

# Parameter aus argparse in Variablen übernehmen
data_preprocessing_mode = args.data_preprocessing_mode
step = args.step
dataset_mode = args.dataset_mode

# Liste der benoetigten Pakete
required_packages = [
    ("numpy", "1.24.4"),
    ("scipy", "1.10.1"),
    ("pandas", "1.5.3"),
    ("matplotlib", None),
    ("scikit-learn", "1.2.2"),
    ("mediapipe", None),
    ("torch", None),
    ("torchvision", None),
    ("torchsummary", None)
]
# Pakete iterativ installieren/importieren
for package_name, version in required_packages:
    install_and_import(package_name, version)

import pandas as pd
import torch
from data_processing import process_or_load_data
from utils_tokenizer import prepare_tokenizers
from data_pipeline import process_train_and_test
from utils_prelstm import load_and_combine_outputs, load_and_combine_labels, prepare_lstm_data
from train import train
from train import save_test_data


USE_CUDA=torch.cuda.is_available()
import torch
import subprocess

# GPU mit dem meisten freien Speicher finden
def get_least_used_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True,
    )
    free_memory = [int(x) for x in result.stdout.strip().split("\n")]
    return free_memory.index(max(free_memory))

gpu_id = get_least_used_gpu()
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Selected GPU: {torch.cuda.get_device_name(device.index)} with ID: {gpu_id}")
else:
    print("CUDA is not available. Using CPU.")

from config import (
    target_count, factor,
    original_train_folder, processed_train_folder, result_train_folder, train_label_csv,
    original_test_folder, processed_test_folder, result_test_folder, test_label_csv,
    num_classes, batch_size, hidden_size, epochs, learning_rate
)

if dataset_mode == 0 or dataset_mode == 1:
    # Verarbeitung der Train-Daten
    print("Processing Train Data...")
    train_files = process_or_load_data(
        data_preprocessing_mode, step, original_train_folder, 
        processed_train_folder, target_count, result_train_folder
    )
    print(f"Train Data: {len(train_files)} files processed.")

if dataset_mode == 0 or dataset_mode == 2:
    # Verarbeitung der Test-Daten
    print("Processing Test Data...")
    test_files = process_or_load_data(
        data_preprocessing_mode, step, original_test_folder, 
        processed_test_folder, target_count, result_test_folder
    )
    print(f"Test Data: {len(test_files)} files processed.")

if dataset_mode == 1 or dataset_mode == 2:
    print("Specific Data processing complete. Please set config.")
    exit()

print("Data processing complete.")

# Tokenizer
finished_dir = "finished"
train_tokenizer, test_tokenizer = prepare_tokenizers(train_label_csv, test_label_csv, finished_dir)

process_train_and_test(train_files, test_files, result_train_folder, result_test_folder, device, factor, num_classes)

##### Test-Daten
print("Verarbeite Test-Daten...")
test_outputs_tensor, test_skipped_files = load_and_combine_outputs(result_test_folder)

test_labels_tensor = load_and_combine_labels(result_test_folder, test_skipped_files)

# Mapping und Tokenisierung für Test-Daten
test_mapping_df = pd.read_csv(test_label_csv, delimiter="|")
test_label_mapping = dict(zip(test_mapping_df['name'], test_mapping_df['translation']))

test_outputs, test_labels = prepare_lstm_data(
    test_outputs_tensor, test_labels_tensor, test_tokenizer, test_label_mapping
)

print(f"Test-Outputs: {test_outputs.shape}, Test-Labels: {test_labels.shape}")

# Test-Daten speichern
save_test_data(test_outputs, test_labels)

# Speicher freigeben
del test_outputs_tensor, test_labels_tensor

##### Train-Daten
print("Verarbeite Train-Daten...")
train_outputs_tensor, train_skipped_files = load_and_combine_outputs(result_train_folder)
train_labels_tensor = load_and_combine_labels(result_train_folder, train_skipped_files)

# Mapping und Tokenisierung für Train-Daten
train_mapping_df = pd.read_csv(train_label_csv, delimiter="|")
train_label_mapping = dict(zip(train_mapping_df['name'], train_mapping_df['translation']))

train_outputs, train_labels = prepare_lstm_data(
    train_outputs_tensor, train_labels_tensor, train_tokenizer, train_label_mapping
)

print(f"Train-Outputs: {train_outputs.shape}, Train-Labels: {train_labels.shape}")

# LSTM-Training und Validierung
train(
    train_outputs=train_outputs,
    train_labels=train_labels,
    vocab_size=len(train_tokenizer.word_to_idx),
    device=device,
    epochs=epochs,
    batch_size=batch_size,
    hidden_size=hidden_size,
    learning_rate=learning_rate
)