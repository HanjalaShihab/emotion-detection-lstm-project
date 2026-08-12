import { useState } from "react";

const EMOTION_COLORS = {
  joy: "#FFD166",
  sadness: "#4C6EF5",
  anger: "#E63946",
  fear: "#9D4EDD",
  love: "#F72585",
  surprise: "#06D6A0",
};

const EMOTION_EMOJI = {
  joy: "😊",
  sadness: "😢",
  anger: "😠",
  fear: "😨",
  love: "❤️",
  surprise: "😲",
};

const IMAGE_EMOTION_COLORS = {
  angry: "#E63946",
  disgusted: "#6A994E",
  fearful: "#9D4EDD",
  happy: "#FFD166",
  neutral: "#94A3B8",
  sad: "#4C6EF5",
  surprised: "#06D6A0",
};

const IMAGE_EMOTION_EMOJI = {
  angry: "😠",
  disgusted: "🤢",
  fearful: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😲",
};

function ChainLoader() {
  // Small signature animation: tokens flowing through LSTM "cells"
  return (
    <div className="flex items-center justify-center gap-2 py-6">
      {[0, 1, 2, 3, 4].map((i) => (
        <span
          key={i}
          className="h-3 w-3 rounded-sm bg-emerald-400"
          style={{
            animation: `pulseCell 1.1s ease-in-out ${i * 0.12}s infinite`,
          }}
        />
      ))}
      <style>{`
        @keyframes pulseCell {
          0%, 100% { opacity: 0.25; transform: scale(0.85); }
          50% { opacity: 1; transform: scale(1.15); }
        }
      `}</style>
    </div>
  );
}

function ProbabilityBar({ label, value, colors = EMOTION_COLORS }) {
  const color = colors[label] ?? "#888";
  const pct = Math.round(value * 100);
  return (
    <div className="mb-2">
      <div className="mb-1 flex justify-between text-xs text-slate-400">
        <span className="capitalize">{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-800">
        <div
          className="h-2 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeEmotion = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        "Could not reach the backend. Make sure FastAPI is running on http://localhost:8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const accentColor = result ? EMOTION_COLORS[result.emotion] : "#3fb950";

  // --- Image emotion detection ---
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setImageResult(null);
    setImageError("");
  };

  const analyzeImage = async () => {
    if (!imageFile) return;

    setImageLoading(true);
    setImageError("");
    setImageResult(null);

    try {
      const formData = new FormData();
      formData.append("file", imageFile);

      const response = await fetch("http://localhost:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Backend request failed");

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setImageResult(data);
    } catch (err) {
      setImageError(err.message || "Could not analyze the image.");
    } finally {
      setImageLoading(false);
    }
  };

  const imageAccentColor = imageResult
    ? IMAGE_EMOTION_COLORS[imageResult.emotion]
    : "#3fb950";

  return (
    <div className="min-h-screen w-full px-4 py-10 text-slate-100">
      <div className="mx-auto max-w-xl">
        {/* Header */}
        <div className="mb-8 text-center">
          <p className="mb-2 font-body text-xs uppercase tracking-[0.3em] text-emerald-400">
            Embedding → LSTM → Dense → Softmax
          </p>
          <h1 className="font-display text-3xl font-bold text-white sm:text-4xl">
            Text Emotion Detection
          </h1>
          <p className="mt-2 font-body text-sm text-slate-400">
            Deep Learning Based Emotion Classification Using LSTM
          </p>
        </div>

        {/* Input card */}
        <div className="rounded-2xl border border-slate-800 bg-[#161b22] p-6 shadow-xl">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter your text..."
            rows={4}
            className="w-full resize-none rounded-lg border border-slate-700 bg-[#0e1117] p-3 font-body text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-emerald-500"
          />

          <button
            onClick={analyzeEmotion}
            disabled={loading || !input.trim()}
            className="mt-4 w-full rounded-lg bg-emerald-500 py-3 font-display font-semibold text-slate-900 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Analyzing..." : "Analyze Emotion"}
          </button>

          {loading && <ChainLoader />}

          {error && (
            <p className="mt-4 text-center text-sm text-red-400">{error}</p>
          )}

          {result && !loading && (
            <div className="mt-6 border-t border-slate-800 pt-6">
              <p className="text-center font-body text-xs uppercase tracking-widest text-slate-500">
                Predicted Emotion
              </p>

              <div
                className="my-3 text-center font-display text-2xl font-bold uppercase"
                style={{ color: accentColor }}
              >
                {EMOTION_EMOJI[result.emotion]} {result.emotion}
              </div>

              <p className="mb-4 text-center text-sm text-slate-400">
                Confidence: {Math.round(result.confidence * 100)}%
              </p>

              {result.probabilities && (
                <div className="mt-4">
                  {Object.entries(result.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([label, value]) => (
                      <ProbabilityBar key={label} label={label} value={value} />
                    ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Image emotion detection card */}
        <div className="mt-8 rounded-2xl border border-slate-800 bg-[#161b22] p-6 shadow-xl">
          <h2 className="mb-1 text-center font-display text-lg font-bold text-white">
            Image Emotion Detection
          </h2>
          <p className="mb-4 text-center font-body text-xs text-slate-400">
            CNN-based facial emotion classification
          </p>

          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="block w-full text-sm text-slate-300 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-700 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-slate-600"
          />

          {imagePreview && (
            <img
              src={imagePreview}
              alt="preview"
              className="mx-auto mt-4 h-32 w-32 rounded-lg border border-slate-700 object-cover"
            />
          )}

          <button
            onClick={analyzeImage}
            disabled={imageLoading || !imageFile}
            className="mt-4 w-full rounded-lg bg-emerald-500 py-3 font-display font-semibold text-slate-900 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {imageLoading ? "Analyzing..." : "Analyze Image"}
          </button>

          {imageLoading && <ChainLoader />}

          {imageError && (
            <p className="mt-4 text-center text-sm text-red-400">{imageError}</p>
          )}

          {imageResult && !imageLoading && (
            <div className="mt-6 border-t border-slate-800 pt-6">
              <p className="text-center font-body text-xs uppercase tracking-widest text-slate-500">
                Predicted Emotion
              </p>

              <div
                className="my-3 text-center font-display text-2xl font-bold uppercase"
                style={{ color: imageAccentColor }}
              >
                {IMAGE_EMOTION_EMOJI[imageResult.emotion]} {imageResult.emotion}
              </div>

              <p className="mb-4 text-center text-sm text-slate-400">
                Confidence: {Math.round(imageResult.confidence * 100)}%
              </p>

              {imageResult.probabilities && (
                <div className="mt-4">
                  {Object.entries(imageResult.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([label, value]) => (
                      <ProbabilityBar
                        key={label}
                        label={label}
                        value={value}
                        colors={IMAGE_EMOTION_COLORS}
                      />
                    ))}
                </div>
              )}
            </div>
          )}
        </div>

        <p className="mt-6 text-center font-body text-xs text-slate-600">
          Backend: FastAPI · Text model: LSTM · Image model: CNN · Frontend: React + Vite
        </p>
      </div>
    </div>
  );
}
