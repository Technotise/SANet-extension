import torch
import torch.nn.utils.rnn as rnn_utils
import glob

def load_and_combine_outputs(output_folder):
    all_outputs = []
    skipped_files = []
    expected_shape = None

    for idx, file in enumerate(glob.glob(f"{output_folder}/combined_outputs/output_*.pt")):
        print(f"Lade Output fuer Datei {idx + 1}: {file}")
        try:
            output = torch.load(file)
            if expected_shape is None:
                expected_shape = output.shape
                print(f"Erwartete Dimensionen festgelegt auf: {expected_shape}")
            if output.shape == expected_shape:
                all_outputs.append(output)
            else:
                print(f"Ueberspringe Datei {file}, da die Dimensionen {output.shape} nicht passen.")
                skipped_files.append(file.replace("output_", "labels_"))
        except Exception as e:
            print(f"Fehler beim Laden der Datei {file}: {e}")
            skipped_files.append(file.replace("output_", "labels_"))

    if len(all_outputs) == 0:
        raise ValueError("Keine gueltigen Outputs gefunden.")

    combined_tensor = torch.cat(all_outputs, dim=0)
    print(f"Kombinierte Outputs: {combined_tensor.shape}")
    return combined_tensor, skipped_files

def load_and_combine_labels(output_folder, skipped_files):
    all_labels = []
    for idx, file in enumerate(glob.glob(f"{output_folder}/combined_outputs/labels_*.pt")):
        if file in skipped_files:
            print(f"Ueberspringe zugehoerige Labels-Datei {file}, da der Output uebersprungen wurde.")
            continue
        print(f"Lade Labels fuer Datei {idx + 1}: {file}")
        try:
            labels = torch.load(file)
            all_labels.extend(labels)
        except Exception as e:
            print(f"Fehler beim Laden der Datei {file}: {e}")

    if len(all_labels) == 0:
        raise ValueError("Keine Labels gefunden.")

    print(f"Anzahl gesammelter Labels: {len(all_labels)}")
    return all_labels

def prepare_lstm_data(outputs, labels, tokenizer, label_mapping):
    # Ordnernamen in Texte umwandeln
    text_labels = [label_mapping[label] for label in labels]
    tokenized_texts = [tokenizer.encode(text) for text in text_labels]

    # Padding für Sequenzlaengen
    label_tensor = rnn_utils.pad_sequence(
        [torch.tensor(seq) for seq in tokenized_texts], batch_first=True, padding_value=0
    )

    print(f"Kombinierter Output fuer LSTM: {outputs.shape}")
    print(f"Geladene Labels: {label_tensor.shape}")
    return outputs, label_tensor

