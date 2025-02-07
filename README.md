# SANet: Skeleton-Aware Neural Sign Language Translation

**Eine reproduzierbare Implementierung und Optimierung**

## 📖 Übersicht

Diese Arbeit baut auf dem **SANet-Modell (Skeleton-Aware Neural Sign Language Translation)** auf und erweitert es durch eine vollständige Implementierung und Optimierung. Ziel ist es, ein leistungsfähiges System zur Übersetzung von Gebärdensprache in Text bereitzustellen, das auf **Skelettdaten aus Videomaterial** basiert.

### Verbesserungen:
- **Synchronisation & Normalisierung der Eingabedaten**
- **Optimierte Batch-Verarbeitung** zur Reduktion von Speicher- und Rechenanforderungen
- **Modulare Code-Struktur** für bessere Erweiterbarkeit und Wartbarkeit
- **Verbesserte BLEU- und ROUGE-Werte** durch optimierte Datenverarbeitung und Architektur

## 📌 Merkmale

✅ **End-to-End-Implementierung** des SANet-Modells  
✅ **Verbesserte Datenverarbeitung** für stabile Trainingsergebnisse  
✅ **GPU-optimierte Batch-Verarbeitung** für effiziente Nutzung der Rechenleistung  
✅ **Erweiterbare & modulare Architektur** für zukünftige Weiterentwicklungen  

## 🛠 Installation

### Voraussetzungen:
- Python 3.8+
- CUDA-fähige GPU (empfohlen)
- [PyTorch](https://pytorch.org/) für neuronale Netze
- OpenPose oder MediaPipe zur Extraktion von Skelettdaten

### Installation:
```bash
git clone https://github.com/Technotise/SANet-extension.git
cd SANet-extension
```

## 🚀 Nutzung

Das Programm wird mit folgendem Befehl ausgeführt:
```bash
python3 main.py
```

### **1. Vorbereitung der Daten**
Einstellungen müssen in der **config.py** durchgeführt werden.

### **2. Mögliche Parameter und deren Anwendung**
Das Skript unterstützt folgende Parameter:

| Parameter | Typ | Mögliche Werte | Standardwert | Beschreibung |
|-----------|------|----------------|--------------|--------------|
| `--data_preprocessing_mode` | str | `process`, `load` | `load` | Wählt den Modus zur Datenvorverarbeitung |
| `--step` | int | 0 - 7 | `0` | Wählt den spezifischen Vorverarbeitungsschritt |
| `--dataset_mode` | int | 0, 1, 2 | `0` | Wählt den Datensatzmodus: 0 = Train & Test, 1 = Train, 2 = Test |

Beispiel für die Nutzung:
```bash
python3 main.py --data_preprocessing_mode process --step 3 --dataset_mode 1
```

## 📊 Ergebnisse

Die implementierten Verbesserungen führten zu signifikanten Verbesserungen der BLEU- und ROUGE-Scores:

| Metrik   | Original SANet | Optimierte Implementierung |
|----------|---------------|---------------------------|
| ROUGE-1  | 0.548         | **0.7317**               |
| ROUGE-2  | 0.573         | **0.7250**               |
| BLEU-1   | 0.573         | **0.5769**               |
| BLEU-2   | 0.424         | **0.5728**               |
| BLEU-3   | 0.322         | **0.5717**               |
| BLEU-4   | 0.248         | **0.5641**               |

## 📚 Weitere Informationen

Diese Arbeit wurde im Rahmen der **Projektgruppe Gebärdensprache** an der **Fachhochschule Südwestfalen** (Sommersemester 2024 – Wintersemester 2024/25) durchgeführt.
