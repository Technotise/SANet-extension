import os
import glob
import pandas as pd
import torch
from torch.cuda.amp import autocast
from utils_model import extend_dataframe_with_skeletons, get_outputs_cliprep, get_outputs_clipscl, run_combined_model

class CombinedModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CombinedModel, self).__init__()
        self.fc_adjust = torch.nn.Linear(4096, 2048)
        self.classifier = torch.nn.Linear(2048, num_classes)

    def forward(self, frames, rgb_skeleton_concat, skeleton_data):
        fc_adjusted = self.fc_adjust(rgb_skeleton_concat)
        fused_features = frames + fc_adjusted
        final_features = fused_features * skeleton_data
        output = self.classifier(final_features)
        return output

import os
import glob
import pandas as pd
import torch

def process_and_apply(data_files, result_folder, device, model, factor):
    print(f"Processing data in folder: {result_folder}")
    intermediate_output_folder = os.path.join(result_folder, "combined_outputs")
    os.makedirs(intermediate_output_folder, exist_ok=True)

    for idx, file in enumerate(data_files):
        print(f"Processing file {idx + 1}/{len(data_files)}: {file}")

        base_filename = os.path.basename(file)
        result_file_final = os.path.join(result_folder, f"final_{base_filename}")

        # Überspringen, wenn die Finaldatei bereits existiert
        if os.path.exists(result_file_final):
            print(f"Final result file for {base_filename} already exists. Skipping...")
            continue

        # Verarbeitungskette
        try:
            df = pd.read_pickle(file)

            # Schritt 1: Skeleton erweitern
            df = extend_dataframe_with_skeletons(df)

            # Schritt 2: Clip-Representation erstellen
            df = get_outputs_cliprep(df, factor)

            # Schritt 3: Clip-Similarity erstellen
            df = get_outputs_clipscl(df, device)

            # Finale Datei speichern
            df.to_pickle(result_file_final)
            print(f"Final result file saved for {base_filename}")
        except Exception as e:
            print(f"Error processing {base_filename}: {e}")
            raise

    print("Running combined model...")
    for idx, file in enumerate(glob.glob(f"{result_folder}/final_*.pkl")):
        print(f"Applying model to file {idx + 1}: {file}")

        base_filename = os.path.basename(file)
        output_file = os.path.join(intermediate_output_folder, f"output_{base_filename}.pt")
        labels_file = os.path.join(intermediate_output_folder, f"labels_{base_filename}.pt")

        # Überspringen, wenn die Modelausgabe bereits existiert
        if os.path.exists(output_file):
            print(f"Output for {base_filename} already exists. Skipping...")
            continue

        # Modelausführung
        try:
            df = pd.read_pickle(file)
            labels, output = run_combined_model(df, model)
            torch.save(output, output_file)
            torch.save(labels, labels_file)
            print(f"Saved outputs and labels for {base_filename}.")
        except Exception as e:
            print(f"Error applying model to {base_filename}: {e}")
            raise

def process_train_and_test(train_files, test_files, result_train_folder, result_test_folder, device, factor, num_classes):
    model = CombinedModel(num_classes=num_classes).to(device)
    print("Processing Train Data...")
    process_and_apply(train_files, result_train_folder, device, model, factor)
    print("Processing Test Data...")
    process_and_apply(test_files, result_test_folder, device, model, factor)