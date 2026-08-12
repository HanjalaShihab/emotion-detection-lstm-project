import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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
  return (
    <div className="flex items-center justify-center gap-2 py-5">
      {[0, 1, 2, 3, 4].map((i) => (
        <span
          key={i}
          className="h-2.5 w-2.5 rounded-sm bg-emerald-400"
          style={{
            animation: `pulseCell 1.1s ease-in-out ${i * 0.12}s infinite`,
          }}
        />
      ))}
      <style>{`
        @keyframes pulseCell {
          0%, 100% {
            opacity: 0.25;
            transform: scale(0.85);
          }
          50% {
            opacity: 1;
            transform: scale(1.15);
          }
        }
      `}</style>
    </div>
  );
}

function ProbabilityBar({ label, value, colors }) {
  const color = colors[label] || "#888";
  const pct = Math.round(value * 100);

  return (
    <div className="mb-3">
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="capitalize text-slate-400">{label}</span>
        <span className="font-medium text-slate-300">{pct}%</span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </div>
  );
}

function ResultCard({
  result,
  colors,
  emojis,
  title = "Predicted Emotion",
}) {
  if (!result) return null;

  const emotion = result.emotion;
  const accentColor = colors[emotion] || "#3fb950";

  return (
    <div className="mt-6 border-t border-slate-800 pt-6">
      <p className="text-center text-xs font-medium uppercase tracking-[0.2em] text-slate-500">
        {title}
      </p>

      <div
        className="mt-3 text-center text-3xl font-bold capitalize"
        style={{ color: accentColor }}
      >
        {emojis[emotion] || "🙂"} {emotion}
      </div>

      <p className="mt-2 text-center text-sm text-slate-400">
        Confidence:{" "}
        <span className="font-semibold text-slate-300">
          {Math.round(result.confidence * 100)}%
        </span>
      </p>

      {result.probabilities && (
        <div className="mt-6">
          {Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([label, value]) => (
              <ProbabilityBar
                key={label}
                label={label}
                value={value}
                colors={colors}
              />
            ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");

  const analyzeEmotion = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: input.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed.");
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearText = () => {
    setInput("");
    setResult(null);
    setError("");
  };

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setImageError("Please select a valid image file.");
      return;
    }

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

      const response = await fetch(`${API_URL}/predict-image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend request failed.");
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setImageResult(data);
    } catch (err) {
      setImageError(
        err.message || "Could not analyze the selected image."
      );
    } finally {
      setImageLoading(false);
    }
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setImageResult(null);
    setImageError("");

    const input = document.getElementById("image-upload");

    if (input) {
      input.value = "";
    }
  };

  return (
    <div className="min-h-screen w-full bg-[#0d1117] px-4 py-8 text-slate-100 sm:px-6 sm:py-12">
      <div className="mx-auto w-full max-w-3xl">
        {/* Header */}
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl md:text-5xl">
            Emotion Detection
          </h1>

          <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-slate-400 sm:text-base">
            Detect emotions from text using LSTM and facial expressions
            using CNN.
          </p>
        </header>

        <div className="space-y-6">
          {/* Text Emotion Card */}
          <section className="rounded-2xl border border-slate-800 bg-[#161b22] p-5 shadow-xl sm:p-7">
            <div className="mb-5">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-lg">
                  💬
                </div>

                <div>
                  <h2 className="text-lg font-bold text-white sm:text-xl">
                    Text Emotion Detection
                  </h2>

                  <p className="text-xs text-slate-500 sm:text-sm">
                    LSTM-based emotion classification
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-slate-800 bg-[#0e1117] px-3 py-2 text-center text-xs text-slate-500">
                Text → Tokenization → Embedding → LSTM → Dense → Softmax
              </div>
            </div>

            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Type something like: I am really happy today!"
              rows={5}
              className="w-full resize-none rounded-xl border border-slate-700 bg-[#0e1117] p-4 text-sm leading-6 text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/30"
            />

            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
              <button
                onClick={analyzeEmotion}
                disabled={loading || !input.trim()}
                className="flex-1 rounded-xl bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Analyzing..." : "Analyze Emotion"}
              </button>

              {input && (
                <button
                  onClick={clearText}
                  disabled={loading}
                  className="rounded-xl border border-slate-700 px-5 py-3 text-sm font-medium text-slate-400 transition hover:border-slate-600 hover:text-white disabled:opacity-40"
                >
                  Clear
                </button>
              )}
            </div>

            {loading && <ChainLoader />}

            {error && (
              <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-center text-sm text-red-400">
                {error}
              </div>
            )}

            {result && !loading && (
              <ResultCard
                result={result}
                colors={EMOTION_COLORS}
                emojis={EMOTION_EMOJI}
              />
            )}
          </section>

          {/* Image Emotion Card */}
          <section className="rounded-2xl border border-slate-800 bg-[#161b22] p-5 shadow-xl sm:p-7">
            <div className="mb-5">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-lg">
                  🖼️
                </div>

                <div>
                  <h2 className="text-lg font-bold text-white sm:text-xl">
                    Image Emotion Detection
                  </h2>

                  <p className="text-xs text-slate-500 sm:text-sm">
                    CNN-based facial emotion classification
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-slate-800 bg-[#0e1117] px-3 py-2 text-center text-xs text-slate-500">
                Image → CNN → Feature Extraction → Dense → Softmax
              </div>
            </div>

            <label
              htmlFor="image-upload"
              className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-700 bg-[#0e1117] px-4 py-8 text-center transition hover:border-emerald-500/50 hover:bg-slate-900/40"
            >
              <span className="text-3xl">📷</span>

              <span className="mt-3 text-sm font-medium text-slate-300">
                Choose an image
              </span>

              <span className="mt-1 text-xs text-slate-600">
                JPG, JPEG, PNG
              </span>

              <input
                id="image-upload"
                type="file"
                accept="image/jpeg,image/png,image/jpg"
                onChange={handleImageChange}
                className="hidden"
              />
            </label>

            {imageFile && (
              <p className="mt-3 truncate text-center text-xs text-slate-500">
                Selected: {imageFile.name}
              </p>
            )}

            {imagePreview && (
              <div className="mt-5 flex justify-center">
                <div className="relative">
                  <img
                    src={imagePreview}
                    alt="Selected preview"
                    className="h-48 w-48 rounded-xl border border-slate-700 object-cover shadow-lg sm:h-56 sm:w-56"
                  />

                  <button
                    onClick={clearImage}
                    disabled={imageLoading}
                    className="absolute -right-2 -top-2 flex h-7 w-7 items-center justify-center rounded-full border border-slate-700 bg-[#161b22] text-sm text-slate-400 transition hover:text-white disabled:opacity-40"
                    aria-label="Remove image"
                  >
                    ×
                  </button>
                </div>
              </div>
            )}

            <div className="mt-5 flex flex-col gap-2 sm:flex-row">
              <button
                onClick={analyzeImage}
                disabled={imageLoading || !imageFile}
                className="flex-1 rounded-xl bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {imageLoading ? "Analyzing..." : "Analyze Image"}
              </button>

              {imageFile && (
                <button
                  onClick={clearImage}
                  disabled={imageLoading}
                  className="rounded-xl border border-slate-700 px-5 py-3 text-sm font-medium text-slate-400 transition hover:border-slate-600 hover:text-white disabled:opacity-40"
                >
                  Clear
                </button>
              )}
            </div>

            {imageLoading && <ChainLoader />}

            {imageError && (
              <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-center text-sm text-red-400">
                {imageError}
              </div>
            )}

            {imageResult && !imageLoading && (
              <ResultCard
                result={imageResult}
                colors={IMAGE_EMOTION_COLORS}
                emojis={IMAGE_EMOTION_EMOJI}
              />
            )}
          </section>
        </div>
      </div>
    </div>
  );
}