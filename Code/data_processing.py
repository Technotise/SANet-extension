import glob
import pandas as pd
from data_preprocess import preprocess
from net_vgg import vggnet_bridge

def process_or_load_data(data_preprocessing_mode, step, original_data_folder, processed_data_folder, target_count, result_save_folder):
    """
    Prozessiert oder laedt Daten, basierend auf dem Modus.
    """
    print(f"Processing mode: {data_preprocessing_mode} for {original_data_folder}")
    processed_files = []

    if data_preprocessing_mode == 'process':
        print("Preprocessing data...")
        preprocess(step, original_data_folder, processed_data_folder, target_count)
        print("Running VGGNet bridge...")
        vggnet_bridge(processed_data_folder, 0.25, "cuda", result_save_folder)
        print("Preprocessing complete.")
    
    print("Loading processed files...")
    processed_files = glob.glob(f"{result_save_folder}/result*.pkl")
    print(f"Found {len(processed_files)} files in {result_save_folder}.")

    return processed_files
