# Text Emotion Detection Using LSTM

A simple university project that classifies text into six emotions — **joy, sadness,
anger, fear, love, surprise** — using a trained LSTM deep-learning model, a FastAPI
backend, and a React frontend.

```
project/
├── training/
│   ├── train.py          # trains the LSTM, saves model + tokenizer
│   ├── requirements.txt
│   └── README.md         # where to get the dataset
├── backend/
│   ├── main.py            # FastAPI app: /health, /predict
│   └── requirements.txt
│   (emotion_model.keras, tokenizer.pkl, label_classes.pkl are created by training)
└── frontend/
    ├── src/App.jsx         # single-page UI
    ├── src/main.jsx
    ├── src/index.css
    └── package.json
```

## How it works

```
React (Vite)  --POST /predict-->  FastAPI  --tokenize+pad-->  LSTM model  --prediction-->  JSON  -->  React
```

The model is trained **once**, offline, by `training/train.py`. The FastAPI backend only
**loads** the saved model and tokenizer — it never trains anything when handling a request.

## Setup & run (3 steps)

### 0. Get the dataset

See `training/README.md` — download the public 6-class emotion dataset and save it as
`training/emotion.csv`.

### 1. Train the model

```bash
cd training
pip install -r requirements.txt
python train.py
```

This creates, inside `backend/`:
- `emotion_model.keras`
- `tokenizer.pkl`
- `label_classes.pkl`

### 2. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs at `http://localhost:8000`. Check `http://localhost:8000/health`.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173`. Open it, type a sentence, click **Analyze Emotion**.

## API

**GET /health**
```json
{ "status": "ok" }
```

**POST /predict**
```json
// request
{ "text": "I am very happy today!" }

// response
{
  "emotion": "joy",
  "emoji": "😊",
  "confidence": 0.92,
  "probabilities": {
    "joy": 0.92,
    "sadness": 0.01,
    "anger": 0.01,
    "fear": 0.02,
    "love": 0.03,
    "surprise": 0.01
  }
}
```

## Notes for explaining to your teacher

- The model is a small, standard architecture: `Embedding → LSTM → Dense → Softmax`.
- Training, saving, and serving are cleanly separated: `train.py` trains and saves;
  `main.py` only loads and predicts.
- The frontend never talks to any AI API directly — it only calls your own FastAPI
  server, which runs the actual LSTM model.
