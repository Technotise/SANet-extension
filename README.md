# SANet: Skeleton-Aware Neural Sign Language Translation

**Eine reproduzierbare Implementierung und Optimierung**

## 📚 Übersicht

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

### Installation der Abhängigkeiten:
```bash
git clone https://github.com/Technotise/SANet-extension.git
cd SANet-extension
pip install -r requirements.txt
```

## 🚀 Nutzung

### **1. Vorbereitung der Daten**
Die Methode benötigt **Skelettdaten aus Videos**. Falls du eigene Daten verwendest, kannst du sie mit **OpenPose oder MediaPipe** extrahieren.

```bash
python preprocess.py --input <video_path> --output <skeleton_data>
```

### **2. Training starten**
Starte das Modelltraining mit:

```bash
python train.py --data <skeleton_data> --epochs 50 --batch_size 32
```

### **3. Modell evaluieren**
Das trainierte Modell kann mit etablierten Metriken bewertet werden:

```bash
python evaluate.py --model <trained_model>
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

