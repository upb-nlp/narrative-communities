"""
LightGBM-based classifier for fake news detection.

Uses text/stylistic features + TF-IDF (word & character n-grams) to classify
articles as real or fake.
"""
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Romanian stop words (most frequent, not discriminative)
# ---------------------------------------------------------------------------
_RO_STOP_WORDS = [
    "de", "și", "în", "a", "la", "că", "cu", "pe", "din", "un", "o", "nu",
    "mai", "este", "se", "ce", "care", "au", "sunt", "fost", "sa", "lui",
    "fie", "fi", "iar", "dar", "pentru", "sau", "al", "ca", "le", "vom",
    "pot", "va", "am", "era", "acest", "această", "acesta", "aceasta",
    "cei", "ale", "ori", "când", "cum", "dacă", "doar", "foarte", "tot",
    "și", "este", "sunt", "prin", "despre", "peste", "între", "după",
    "până", "fără", "astfel", "fiind", "cele", "lor", "ani", "unei",
    "unor", "într", "avea", "avea", "acestui", "aceste", "acești",
]

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
from narrative_map.config import DATASET_NAME, LIGHTGBM_MODEL_PATH, LIGHTGBM_SCALER_PATH, REAL_NEWS_LABEL


# ===================================================================
# Feature dataclass  (text-only — graph features removed)
# ===================================================================
@dataclass
class FeatureSet:
    """Hand-crafted features extracted from headline + body text."""

    # --- text structure ---
    headline_length: int = 0
    headline_word_count: int = 0
    body_length: int = 0
    body_word_count: int = 0
    avg_word_length: float = 0.0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    paragraph_count: int = 0
    avg_paragraph_length: float = 0.0
    type_token_ratio: float = 0.0
    headline_body_overlap: float = 0.0

    # --- style indicators ---
    exclamation_count: int = 0
    question_count: int = 0
    uppercase_ratio: float = 0.0
    caps_word_count: int = 0
    quote_count: int = 0
    number_count: int = 0
    punctuation_density: float = 0.0

    # --- credibility signals ---
    has_quotes: int = 0
    has_numbers: int = 0
    has_source_attribution: int = 0

    # --- emotional / sensational ---
    emotional_word_count: int = 0
    urgency_word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "headline_length": self.headline_length,
            "headline_word_count": self.headline_word_count,
            "body_length": self.body_length,
            "body_word_count": self.body_word_count,
            "avg_word_length": self.avg_word_length,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": self.avg_sentence_length,
            "paragraph_count": self.paragraph_count,
            "avg_paragraph_length": self.avg_paragraph_length,
            "type_token_ratio": self.type_token_ratio,
            "headline_body_overlap": self.headline_body_overlap,
            "exclamation_count": self.exclamation_count,
            "question_count": self.question_count,
            "uppercase_ratio": self.uppercase_ratio,
            "caps_word_count": self.caps_word_count,
            "quote_count": self.quote_count,
            "number_count": self.number_count,
            "punctuation_density": self.punctuation_density,
            "has_quotes": self.has_quotes,
            "has_numbers": self.has_numbers,
            "has_source_attribution": self.has_source_attribution,
            "emotional_word_count": self.emotional_word_count,
            "urgency_word_count": self.urgency_word_count,
        }


# ===================================================================
# Classifier
# ===================================================================
class LightGBMClassifier:
    """LightGBM classifier for fake news detection."""

    TFIDF_WORD_MAX = 200
    TFIDF_CHAR_MAX = 50

    def __init__(
        self,
        model_path: Path = LIGHTGBM_MODEL_PATH,
        scaler_path: Path = LIGHTGBM_SCALER_PATH,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler: StandardScaler | None = None
        self.tfidf: TfidfVectorizer | None = None
        self.char_tfidf: TfidfVectorizer | None = None
        self.is_trained = False
        self.feature_names: list[str] | None = None
        self.threshold: float = 0.5

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, headline: str, body: str) -> FeatureSet:
        """Extract text / style features from headline and body."""
        text = f"{headline} {body}" if body else headline
        words = text.split() if text else []
        word_count = len(words)

        # Type-token ratio
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / max(word_count, 1)

        # Sentences
        sentences = re.split(r"[.!?]+", body) if body else []
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Paragraphs
        paragraphs = (
            [p.strip() for p in body.split("\n\n") if p.strip()] if body else []
        )
        if not paragraphs and body:
            paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
        paragraph_count = len(paragraphs)
        para_words = [len(p.split()) for p in paragraphs] if paragraphs else [0]
        avg_para_len = sum(para_words) / max(len(paragraphs), 1)

        # Headline / body word overlap (Jaccard)
        h_words = set(headline.lower().split()) if headline else set()
        b_words = set(body.lower().split()) if body else set()
        if h_words and b_words:
            hb_overlap = len(h_words & b_words) / len(h_words | b_words)
        else:
            hb_overlap = 0.0

        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        avg_sent_len = word_count / max(sentence_count, 1)

        # Style
        excl = text.count("!")
        ques = text.count("?")
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        upper_ratio = len(caps_words) / max(word_count, 1)
        quote_chars = text.count('"') + text.count("\u201e") + text.count("\u201d")
        nums = len(re.findall(r"\d+", text))

        # Punctuation density
        punct_count = sum(
            1 for c in text if c in '.,;:!?-\u2013\u2014()[]{}"\'\u201e\u201d\u201c\xab\xbb'
        )
        punct_density = punct_count / max(len(text), 1)

        # Credibility (Romanian patterns)
        src_pats = [
            r"\ba declarat\b", r"\ba spus\b", r"\bpotrivit\b",
            r"\bconform\b", r"\bsusține\b", r"\bafirmă\b",
        ]
        has_src = any(re.search(p, text.lower()) for p in src_pats)

        # Emotional words
        emo_words = [
            "șocant", "incredibil", "uimitor", "teribil", "groaznic",
            "fantastic", "minunat", "dezastru", "catastrofă", "tragedie",
            "scandal", "bombă", "exploziv", "cutremurător", "halucinant",
        ]
        emo_count = sum(1 for w in emo_words if w in text.lower())

        # Urgency words
        urg_words = [
            "urgent", "breaking", "acum", "imediat", "alertă",
            "ultima oră", "de ultimă oră", "atenție",
        ]
        urg_count = sum(1 for w in urg_words if w in text.lower())

        return FeatureSet(
            headline_length=len(headline) if headline else 0,
            headline_word_count=len(headline.split()) if headline else 0,
            body_length=len(body) if body else 0,
            body_word_count=len(body.split()) if body else 0,
            avg_word_length=avg_word_len,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sent_len,
            paragraph_count=paragraph_count,
            avg_paragraph_length=avg_para_len,
            type_token_ratio=ttr,
            headline_body_overlap=hb_overlap,
            exclamation_count=excl,
            question_count=ques,
            uppercase_ratio=upper_ratio,
            caps_word_count=len(caps_words),
            quote_count=quote_chars // 2,
            number_count=nums,
            punctuation_density=punct_density,
            has_quotes=1 if quote_chars > 0 else 0,
            has_numbers=1 if nums > 0 else 0,
            has_source_attribution=1 if has_src else 0,
            emotional_word_count=emo_count,
            urgency_word_count=urg_count,
        )

    def extract_features_from_dict(
        self, _features_dict: dict, headline: str, body: str
    ) -> FeatureSet:
        """Backward-compatible wrapper; graph fields in *_features_dict* are ignored."""
        return self.extract_features(headline, body)

    # ------------------------------------------------------------------
    # TF-IDF helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_col_names(prefix: str, raw_names) -> list[str]:
        """Sanitize and deduplicate TF-IDF column names for LightGBM."""
        seen: dict[str, int] = {}
        cols: list[str] = []
        for raw in raw_names:
            clean = re.sub(r"[^\w]", "_", raw, flags=re.UNICODE).strip("_")
            key = f"{prefix}_{clean}"
            if key in seen:
                seen[key] += 1
                key = f"{key}_{seen[key]}"
            else:
                seen[key] = 0
            cols.append(key)
        return cols

    @staticmethod
    def _build_tfidf_text(headline: str, body: str) -> str:
        parts = []
        if headline:
            parts.append(headline)
        if body:
            parts.append(body[:3000])
        return " ".join(parts).lower()

    def _fit_word_tfidf(self, texts: list[str]) -> pd.DataFrame:
        self.tfidf = TfidfVectorizer(
            max_features=self.TFIDF_WORD_MAX,
            ngram_range=(1, 2),
            stop_words=_RO_STOP_WORDS,
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
        )
        mat = self.tfidf.fit_transform(texts)
        cols = self._safe_col_names("tfidf", self.tfidf.get_feature_names_out())
        return pd.DataFrame(mat.toarray(), columns=cols)

    def _transform_word_tfidf(self, texts: list[str]) -> pd.DataFrame:
        assert self.tfidf is not None
        mat = self.tfidf.transform(texts)
        cols = self._safe_col_names("tfidf", self.tfidf.get_feature_names_out())
        return pd.DataFrame(mat.toarray(), columns=cols)

    def _fit_char_tfidf(self, texts: list[str]) -> pd.DataFrame:
        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self.TFIDF_CHAR_MAX,
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
        )
        mat = self.char_tfidf.fit_transform(texts)
        cols = self._safe_col_names("chartf", self.char_tfidf.get_feature_names_out())
        return pd.DataFrame(mat.toarray(), columns=cols)

    def _transform_char_tfidf(self, texts: list[str]) -> pd.DataFrame:
        assert self.char_tfidf is not None
        mat = self.char_tfidf.transform(texts)
        cols = self._safe_col_names("chartf", self.char_tfidf.get_feature_names_out())
        return pd.DataFrame(mat.toarray(), columns=cols)

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------
    def prepare_features(self, feature_sets: list[FeatureSet]) -> pd.DataFrame:
        df = pd.DataFrame([fs.to_dict() for fs in feature_sets])
        self.feature_names = df.columns.tolist()
        return df

    def _build_inference_df(
        self, features: FeatureSet, headline: str = "", body: str = ""
    ) -> pd.DataFrame:
        """Single-row DataFrame aligned to training feature order."""
        df = pd.DataFrame([features.to_dict()])

        if self.tfidf is not None or self.char_tfidf is not None:
            text = self._build_tfidf_text(headline, body)
            if self.tfidf is not None:
                df = pd.concat(
                    [df.reset_index(drop=True), self._transform_word_tfidf([text])],
                    axis=1,
                )
            if self.char_tfidf is not None:
                df = pd.concat(
                    [df.reset_index(drop=True), self._transform_char_tfidf([text])],
                    axis=1,
                )

        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        raw_texts: list[str] | None = None,
        tune_hyperparams: bool = False,
        n_trials: int = 20,
        verbose: bool = True,
    ):
        """Train on all supplied data.

        1. Fit TF-IDF (word + char) and concatenate.
        2. Optuna hyper-parameter search via 5-fold CV on **all** data.
        3. Threshold tuning via out-of-fold predictions.
        4. Refit final model on **all** data.

        Returns (oof_accuracy, oof_f1, oof_auc).
        """
        # --- 1. TF-IDF ---------------------------------------------------
        if raw_texts is not None:
            if verbose:
                print(
                    f"\n  Fitting word TF-IDF ({self.TFIDF_WORD_MAX} features, "
                    f"unigrams+bigrams)..."
                )
            word_df = self._fit_word_tfidf(raw_texts)
            if verbose:
                print(
                    f"  Fitting char TF-IDF ({self.TFIDF_CHAR_MAX} features, "
                    f"3–5 char n-grams)..."
                )
            char_df = self._fit_char_tfidf(raw_texts)
            X = pd.concat(
                [X.reset_index(drop=True), word_df, char_df], axis=1
            )

        self.feature_names = X.columns.tolist()

        if verbose:
            print(f"\n  Training data summary")
            print(f"    Features : {X.shape[1]}")
            print(f"    Samples  : {X.shape[0]}")
            print(f"    Real     : {int((y == 1).sum())}")
            print(f"    Fake     : {int((y == 0).sum())}")

        # --- Scale --------------------------------------------------------
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns
        )

        num_cores = os.cpu_count() or 4
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # --- 2. Optuna (CV on ALL data) -----------------------------------
        if tune_hyperparams and OPTUNA_AVAILABLE:
            if verbose:
                print(
                    f"\n  Optuna hyper-parameter search "
                    f"({n_trials} trials, 5-fold CV on ALL data)..."
                )

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "lambda_l1": trial.suggest_float(
                        "lambda_l1", 1e-3, 10.0, log=True
                    ),
                    "lambda_l2": trial.suggest_float(
                        "lambda_l2", 1e-3, 10.0, log=True
                    ),
                    "feature_fraction": trial.suggest_float(
                        "feature_fraction", 0.4, 1.0
                    ),
                    "bagging_fraction": trial.suggest_float(
                        "bagging_fraction", 0.4, 1.0
                    ),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_data_in_leaf": trial.suggest_int(
                        "min_data_in_leaf", 5, 50
                    ),
                    # fixed
                    "objective": "binary",
                    "boosting_type": "gbdt",
                    "class_weight": "balanced",
                    "num_threads": num_cores,
                    "device": "cpu",
                    "verbose": -1,
                    "random_state": 42,
                }
                m = lgb.LGBMClassifier(**params)
                return cross_val_score(
                    m, X_scaled, y, cv=cv, scoring="roc_auc"
                ).mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective, n_trials=n_trials, show_progress_bar=verbose
            )

            best_params = study.best_params
            if verbose:
                print(f"\n  Best CV AUC: {study.best_value:.4f}")
                print(f"  Best params: {best_params}")
        else:
            best_params = {
                "n_estimators": 300,
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.05,
                "lambda_l1": 0.1,
                "lambda_l2": 1.0,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": 20,
            }

        # Separate n_estimators (not a LightGBM core param dict key)
        n_estimators = best_params.pop("n_estimators", 300)

        # Merge fixed training params
        best_params.update(
            {
                "objective": "binary",
                "boosting_type": "gbdt",
                "class_weight": "balanced",
                "num_threads": num_cores,
                "device": "cpu",
                "verbose": -1,
                "random_state": 42,
            }
        )

        # --- 3. Threshold tuning via OOF predictions ---------------------
        if verbose:
            print("\n  Tuning classification threshold (OOF predictions)...")

        thresh_model = lgb.LGBMClassifier(
            **best_params, n_estimators=n_estimators
        )
        oof_proba = cross_val_predict(
            thresh_model, X_scaled, y, cv=cv, method="predict_proba"
        )[:, 1]

        best_threshold = 0.5
        best_f1 = 0.0
        for t in np.arange(0.30, 0.70, 0.005):
            preds = (oof_proba >= t).astype(int)
            current_f1 = f1_score(y, preds)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(t)
        self.threshold = best_threshold

        # Report OOF metrics
        oof_preds = (oof_proba >= self.threshold).astype(int)
        oof_accuracy = accuracy_score(y, oof_preds)
        oof_f1 = f1_score(y, oof_preds)
        oof_auc = roc_auc_score(y, oof_proba)

        if verbose:
            print(f"    Optimal threshold: {self.threshold:.3f}")
            print(f"\n  Out-of-fold metrics (5-fold CV on ALL data):")
            print(f"    Accuracy: {oof_accuracy:.4f}")
            print(f"    F1-Score: {oof_f1:.4f}")
            print(f"    AUC-ROC:  {oof_auc:.4f}")

        # --- 4. Final model on ALL data -----------------------------------
        if verbose:
            print(
                f"\n  Training final model on all {X_scaled.shape[0]} "
                f"samples (n_estimators={n_estimators})..."
            )

        self.model = lgb.LGBMClassifier(
            **best_params, n_estimators=n_estimators
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        if verbose:
            print(
                f"  ✓ Done  (threshold={self.threshold:.3f})"
            )
            print(f"\n  Feature importance (top 15):")
            importance = (
                pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": self.model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
            )
            for _, row in importance.head(15).iterrows():
                print(f"    {row['feature']}: {row['importance']:.0f}")

        return oof_accuracy, oof_f1, oof_auc

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self, features: FeatureSet, headline: str = "", body: str = ""
    ) -> tuple[str, float]:
        """Predict a single article. Returns (label, confidence)."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        df = self._build_inference_df(features, headline, body)
        df_scaled = pd.DataFrame(
            self.scaler.transform(df), columns=df.columns
        )
        proba = self.model.predict_proba(df_scaled)[0]
        pred_class = 1 if proba[1] >= self.threshold else 0
        classification = "real" if pred_class == 1 else "fake"
        confidence = proba[1] if pred_class == 1 else proba[0]
        return classification, float(confidence)

    def predict_batch(
        self, feature_sets: list[FeatureSet]
    ) -> list[tuple[str, float]]:
        """Predict a batch (hand-crafted features only, no TF-IDF)."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        df = self.prepare_features(feature_sets)
        if self.feature_names and list(df.columns) != self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        df_scaled = pd.DataFrame(
            self.scaler.transform(df), columns=df.columns
        )
        probas = self.model.predict_proba(df_scaled)
        results = []
        for proba in probas:
            pred_class = 1 if proba[1] >= self.threshold else 0
            classification = "real" if pred_class == 1 else "fake"
            confidence = proba[1] if pred_class == 1 else proba[0]
            results.append((classification, float(confidence)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "tfidf": self.tfidf,
                    "char_tfidf": self.char_tfidf,
                    "threshold": self.threshold,
                },
                f,
            )
        print(f"  ✓ Model  → {self.model_path}")
        print(f"  ✓ Scaler → {self.scaler_path}")

    def load(self) -> bool:
        if not self.model_path.exists() or not self.scaler_path.exists():
            return False
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, "rb") as f:
                data = pickle.load(f)
                self.scaler = data["scaler"]
                self.feature_names = data["feature_names"]
                self.tfidf = data.get("tfidf")
                self.char_tfidf = data.get("char_tfidf")
                self.threshold = data.get("threshold", 0.5)
            self.is_trained = True

            parts = [f"Model loaded ({len(self.feature_names)} features"]
            if self.tfidf:
                parts.append(
                    f", word-tfidf={len(self.tfidf.get_feature_names_out())}"
                )
            if self.char_tfidf:
                parts.append(
                    f", char-tfidf={len(self.char_tfidf.get_feature_names_out())}"
                )
            parts.append(f", threshold={self.threshold:.3f})")
            print(f"  ✓ {''.join(parts)}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            return False


# ===================================================================
# Training entry-point
# ===================================================================
def train_classifier(
    tune_hyperparams: bool = False,
    n_trials: int = 20,
    *,
    include_validation: bool = False,
):
    """Train LightGBM on the train split (validation optional via flag)."""
    print("=" * 60)
    print("LightGBM Classifier Training")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    ds = load_dataset(DATASET_NAME, "default")

    samples: list[dict] = []
    if "train" in ds:
        samples.extend(ds["train"].to_list())
    if include_validation and "validation" in ds:
        samples.extend(ds["validation"].to_list())
    if not samples:
        samples = list(ds.values())[0].to_list()
    split_desc = "train+validation" if include_validation else "train"
    print(f"  Samples ({split_desc}): {len(samples)}")

    # Extract features
    classifier = LightGBMClassifier()
    print("\nExtracting features...")

    feature_sets: list[FeatureSet] = []
    labels: list[int] = []
    raw_texts: list[str] = []

    for sample in tqdm(samples, desc="Extracting"):
        headline = sample.get("headline", "")
        body = sample.get("body", "")
        label = 1 if sample.get("supernarrative", "").strip() == REAL_NEWS_LABEL else 0

        fs = classifier.extract_features(headline, body)
        feature_sets.append(fs)
        labels.append(label)
        raw_texts.append(LightGBMClassifier._build_tfidf_text(headline, body))

    print(f"  Real: {sum(labels)}, Fake: {len(labels) - sum(labels)}")

    X = classifier.prepare_features(feature_sets)
    y = np.array(labels)

    classifier.train(
        X,
        y,
        raw_texts=raw_texts,
        tune_hyperparams=tune_hyperparams,
        n_trials=n_trials,
    )
    classifier.save()
    return classifier


# Keep old name as alias for backward compatibility
train_classifier_on_validation = train_classifier


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM classifier")
    parser.add_argument(
        "--tune", action="store_true", help="Tune hyperparameters with Optuna"
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Also include the HF validation split in training data",
    )
    args = parser.parse_args()

    clf = train_classifier(
        tune_hyperparams=args.tune,
        n_trials=args.trials,
        include_validation=args.include_validation,
    )
    print(f"\n✓ Training complete!")
    print(f"  Model : {clf.model_path}")
    print(f"  Scaler: {clf.scaler_path}")
