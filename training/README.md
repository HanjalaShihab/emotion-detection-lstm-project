# Dataset

`train.py` expects a file called **`emotion.csv`** in this folder, with two columns:

```
text,emotion
```

where `emotion` is one of: `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`.

## Recommended dataset

Use the public **"Emotions" dataset** (based on the dair-ai emotion dataset), which already
uses exactly these six labels. You can get it from either source:

- Kaggle: search **"Emotions dataset for NLP"** or **"nelgiriyewithana/emotions"**
  (https://www.kaggle.com/datasets/nelgiriyewithana/emotions)
- Hugging Face: `dair-ai/emotion` (https://huggingface.co/datasets/dair-ai/emotion)

Download the CSV, rename it to `emotion.csv`, and place it in this `training/` folder so
the final layout looks like:

```
training/
├── train.py
├── emotion.csv     <-- put the downloaded dataset here
└── README.md
```

If the file you download has different column names (e.g. `Text`, `Emotion`, or numeric
labels), just rename the columns to `text` and `emotion`, and map any numeric labels back
to the six words above, before running `train.py`.

## Run training

```bash
cd training
python train.py
```

This will:
1. Load and clean `emotion.csv`
2. Split into train/test sets
3. Tokenize and pad the text
4. Train a small LSTM model
5. Print test accuracy/loss and save `training_history.png`
6. Save the trained model + tokenizer + label list into `../backend/`
