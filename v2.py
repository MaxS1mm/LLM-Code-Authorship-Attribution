import json
import math
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

SAMPLE_PER_MODEL = 2000
DATA_PATH = "/Users/maxsimanonok/Desktop/Proj491-2/FormAI-v2.json"

# ── 1. Load & sample ────────────────────────────────────────────────

print("Loading dataset …")
with open(DATA_PATH) as f:
    raw = json.load(f)

by_model = {}
for entry in raw:
    model = entry["file_name"].rsplit("-", 1)[0]
    by_model.setdefault(model, []).append(entry)

del raw  # free memory

rng = np.random.default_rng(42)
sampled = []
for model, entries in by_model.items():
    idxs = rng.choice(len(entries), size=SAMPLE_PER_MODEL, replace=False)
    for i in idxs:
        sampled.append((model, entries[i]))
print(f"Sampled {len(sampled)} entries across {len(by_model)} models")

# ── 2. Feature extraction ───────────────────────────────────────────

def shannon_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

def extract_features(code: str) -> dict:
    lines = code.split("\n")
    num_lines = len(lines)
    line_lengths = [len(l) for l in lines]
    non_empty = [l for l in lines if l.strip()]

    # Whitespace features
    leading_spaces = [len(l) - len(l.lstrip(" ")) for l in lines]
    leading_tabs = [l.count("\t") for l in lines if l.startswith("\t")]
    total_spaces = code.count(" ")
    total_tabs = code.count("\t")

    # Import / include features
    include_lines = [l for l in lines if re.match(r"\s*#\s*include", l)]
    import_count = len(include_lines)

    # Comment features
    single_comments = sum(1 for l in lines if re.search(r"//", l))
    block_open = code.count("/*")
    block_close = code.count("*/")

    # Brace / paren style
    open_braces = code.count("{")
    close_braces = code.count("}")
    open_parens = code.count("(")
    close_parens = code.count(")")
    semicolons = code.count(";")

    # Identifier-style heuristics
    camel = len(re.findall(r"[a-z][A-Z]", code))
    snake = len(re.findall(r"[a-z]_[a-z]", code))

    # Blank line patterns
    blank_lines = sum(1 for l in lines if not l.strip())
    consecutive_blanks = len(re.findall(r"\n\n\n", code))

    # Keyword density
    keywords_c = [
        "if", "else", "for", "while", "do", "switch", "case", "return",
        "break", "continue", "typedef", "struct", "enum", "void", "int",
        "char", "float", "double", "long", "unsigned", "const", "static",
        "sizeof", "malloc", "free", "printf", "scanf", "NULL",
    ]
    word_tokens = re.findall(r"\b[A-Za-z_]\w*\b", code)
    total_tokens = len(word_tokens) or 1
    token_counts = Counter(word_tokens)
    keyword_freq = {f"kw_{kw}": token_counts.get(kw, 0) / total_tokens for kw in keywords_c}

    # String literal counts
    string_lits = len(re.findall(r'"[^"]*"', code))

    # Unique tokens ratio
    unique_ratio = len(set(word_tokens)) / total_tokens

    safe_lines = num_lines or 1
    safe_ne = len(non_empty) or 1

    feats = {
        "num_lines": num_lines,
        "num_chars": len(code),
        "entropy": shannon_entropy(code),
        "mean_line_len": np.mean(line_lengths),
        "median_line_len": np.median(line_lengths),
        "std_line_len": np.std(line_lengths),
        "max_line_len": max(line_lengths) if line_lengths else 0,
        "min_line_len": min(line_lengths) if line_lengths else 0,
        "mean_indent": np.mean(leading_spaces),
        "std_indent": np.std(leading_spaces),
        "max_indent": max(leading_spaces) if leading_spaces else 0,
        "tab_count": total_tabs,
        "space_count": total_spaces,
        "tab_space_ratio": total_tabs / (total_spaces + 1),
        "import_count": import_count,
        "import_ratio": import_count / safe_lines,
        "single_comment_count": single_comments,
        "block_comment_count": block_open,
        "comment_ratio": (single_comments + block_open) / safe_lines,
        "open_braces": open_braces,
        "brace_ratio": open_braces / safe_lines,
        "semicolon_ratio": semicolons / safe_lines,
        "paren_ratio": open_parens / safe_lines,
        "camel_case_count": camel,
        "snake_case_count": snake,
        "camel_snake_ratio": camel / (snake + 1),
        "blank_line_ratio": blank_lines / safe_lines,
        "consecutive_blank_ratio": consecutive_blanks / safe_lines,
        "string_literal_count": string_lits,
        "string_literal_ratio": string_lits / safe_lines,
        "unique_token_ratio": unique_ratio,
        "total_tokens": total_tokens,
        "tokens_per_line": total_tokens / safe_lines,
        "cyclomatic_proxy": open_braces + open_parens,
    }
    feats.update(keyword_freq)
    return feats


print("Extracting features …")
rows = []
labels = []
for model, entry in sampled:
    code = entry.get("source_code", "") or ""
    feats = extract_features(code)
    rows.append(feats)
    labels.append(model)

df = pd.DataFrame(rows)
df["model"] = labels

# Save sampled dataset
df.to_csv("/Users/maxsimanonok/Desktop/Proj491-2/sampled_dataset.csv", index=False)
print(f"Saved sampled_dataset.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")

# ── 3. Train XGBoost ────────────────────────────────────────────────

le = LabelEncoder()
y = le.fit_transform(df["model"])
X = df.drop(columns=["model"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print("Training XGBoost …")

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class=len(le.classes_),
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# ── 4. Evaluate ─────────────────────────────────────────────────────

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print("=" * 60)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

# ── 5. Feature importance ───────────────────────────────────────────

importances = clf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nTop 20 Features by Importance:")
print(feat_imp.head(20).to_string())
