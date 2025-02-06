# config.py

# Globale Konfigurationsvariablen
# data_preprocessing_mode = 'process'  # 'process' oder 'load'
# step = 6 # preprocess Schritt
# dataset_mode = 0 # 0 = Train und Test; 1 = Train; 2 = Test
target_count = 200
factor = 0.25

# Verzeichnisse für die Train-Daten
original_train_folder = "/XYZ/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
processed_train_folder = "/XYZ/preprocessed/train"
result_train_folder = "/XYZ/results/train"
train_label_csv = "/XYZ/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"

# Verzeichnisse für die Test-Daten
original_test_folder = "/XYZ/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test"
processed_test_folder = "/XYZ/preprocessed/test"
result_test_folder = "/XYZ/results/test"
test_label_csv = "/XYZ/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"

# Trainingsparameter
num_classes = 10
batch_size = 64
hidden_size = 512
epochs = 10
learning_rate = 1e-3