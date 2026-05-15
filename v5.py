"""
Enhanced ensemble: XGBoost (stylometric+AST) + TF-IDF LR + CodeBERT embeddings.
Reads the sampled dataset from the previous pipeline and adds 768-dim CLS embeddings.
"""
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb

SAMPLE_PER_MODEL = 5000
DATA_PATH = "/Users/maxsimanonok/Desktop/Proj491-2/FormAI-v2.json"
RANDOM_STATE = 42
BATCH_SIZE = 16
MAX_TOKEN_LEN = 256
DEVICE = "cpu"
EMB_CACHE_TRAIN = "/Users/maxsimanonok/Desktop/Proj491-2/emb_train.npy"
EMB_CACHE_TEST = "/Users/maxsimanonok/Desktop/Proj491-2/emb_test.npy"
EMB_CACHE_LABELS = "/Users/maxsimanonok/Desktop/Proj491-2/emb_train_labels.npy"

# ── 1. Load & sample ────────────────────────────────────────────────

print("Loading dataset …")
with open(DATA_PATH) as f:
    raw = json.load(f)

by_model = {}
for entry in raw:
    model = entry["file_name"].rsplit("-", 1)[0]
    by_model.setdefault(model, []).append(entry)
del raw

rng = np.random.default_rng(RANDOM_STATE)
sampled = []
for model, entries in by_model.items():
    idxs = rng.choice(len(entries), size=SAMPLE_PER_MODEL, replace=False)
    for i in idxs:
        sampled.append((model, entries[i]))
del by_model
import gc; gc.collect()
print(f"Sampled {len(sampled)} entries across 9 models")

# ── 2. Strip model identifiers ──────────────────────────────────────

def strip_model_identifiers(code: str) -> str:
    lines = code.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        if i < 3 and re.search(
            r"(GPT-4o-mini|GPT-3\.5|Falcon-180B|Falcon2-11B|Gemini Pro|Gemma|"
            r"CodeLlama|Llama 2|Mistral|DATASET\s+v\d)",
            line, re.IGNORECASE
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

# ── 3. Feature extraction (same as before) ──────────────────────────

def shannon_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

def extract_ast_features(code):
    lines = code.split("\n")
    func_defs = re.findall(r"^\s*\w[\w\s\*]+\s+(\w+)\s*\([^)]*\)\s*\{", code, re.MULTILINE)
    num_functions = len(func_defs)
    func_name_lengths = [len(n) for n in func_defs] if func_defs else [0]
    max_depth = 0
    cur_depth = 0
    depth_hist = []
    for ch in code:
        if ch == "{":
            cur_depth += 1
            if cur_depth > max_depth:
                max_depth = cur_depth
        elif ch == "}":
            depth_hist.append(cur_depth)
            cur_depth = max(0, cur_depth - 1)
    mean_depth = np.mean(depth_hist) if depth_hist else 0
    if_count = len(re.findall(r"\bif\s*\(", code))
    else_count = len(re.findall(r"\belse\b", code))
    for_count = len(re.findall(r"\bfor\s*\(", code))
    while_count = len(re.findall(r"\bwhile\s*\(", code))
    switch_count = len(re.findall(r"\bswitch\s*\(", code))
    case_count = len(re.findall(r"\bcase\b", code))
    ternary_count = code.count("?") - code.count("\\?")
    goto_count = len(re.findall(r"\bgoto\b", code))
    control_kw = {"if", "for", "while", "switch", "return", "sizeof", "typedef"}
    all_calls = re.findall(r"\b(\w+)\s*\(", code)
    func_calls = [c for c in all_calls if c not in control_kw and c not in func_defs]
    num_func_calls = len(func_calls)
    unique_func_calls = len(set(func_calls))
    struct_defs = len(re.findall(r"\bstruct\s+\w+\s*\{", code))
    enum_defs = len(re.findall(r"\benum\s+\w+\s*\{", code))
    typedef_count = len(re.findall(r"\btypedef\b", code))
    pointer_decls = len(re.findall(r"\w+\s*\*\s*\w+", code))
    arrow_ops = code.count("->")
    deref_count = len(re.findall(r"\*\w+", code))
    array_decls = len(re.findall(r"\w+\s*\[[^\]]*\]", code))
    defines = len(re.findall(r"^\s*#\s*define\b", code, re.MULTILINE))
    ifdefs = len(re.findall(r"^\s*#\s*if", code, re.MULTILINE))
    pragmas = len(re.findall(r"^\s*#\s*pragma\b", code, re.MULTILINE))
    return_count = len(re.findall(r"\breturn\b", code))
    func_starts = [m.start() for m in re.finditer(r"^\s*\w[\w\s\*]+\s+\w+\s*\([^)]*\)\s*\{", code, re.MULTILINE)]
    func_lengths = []
    for start in func_starts:
        depth = 0
        begun = False
        end = start
        for i in range(start, len(code)):
            if code[i] == "{":
                depth += 1
                begun = True
            elif code[i] == "}":
                depth -= 1
                if begun and depth == 0:
                    end = i
                    break
        func_lengths.append(code[start:end].count("\n"))
    avg_func_len = np.mean(func_lengths) if func_lengths else 0
    std_func_len = np.std(func_lengths) if func_lengths else 0
    max_func_len = max(func_lengths) if func_lengths else 0
    param_matches = re.findall(r"\w+\s*\(([^)]*)\)\s*\{", code)
    param_counts = []
    for pm in param_matches:
        pm = pm.strip()
        if pm and pm != "void":
            param_counts.append(pm.count(",") + 1)
        else:
            param_counts.append(0)
    avg_params = np.mean(param_counts) if param_counts else 0
    max_paren_depth = 0
    pd_ = 0
    for ch in code:
        if ch == "(":
            pd_ += 1
            if pd_ > max_paren_depth:
                max_paren_depth = pd_
        elif ch == ")":
            pd_ = max(0, pd_ - 1)
    safe_lines = len(lines) or 1
    safe_funcs = num_functions or 1
    return {
        "ast_num_functions": num_functions,
        "ast_mean_func_name_len": np.mean(func_name_lengths),
        "ast_std_func_name_len": np.std(func_name_lengths),
        "ast_max_nesting_depth": max_depth,
        "ast_mean_nesting_depth": mean_depth,
        "ast_if_count": if_count, "ast_else_count": else_count,
        "ast_for_count": for_count, "ast_while_count": while_count,
        "ast_switch_count": switch_count, "ast_case_count": case_count,
        "ast_ternary_count": ternary_count, "ast_goto_count": goto_count,
        "ast_if_else_ratio": else_count / (if_count + 1),
        "ast_branch_density": (if_count + switch_count) / safe_lines,
        "ast_loop_density": (for_count + while_count) / safe_lines,
        "ast_num_func_calls": num_func_calls,
        "ast_unique_func_calls": unique_func_calls,
        "ast_call_diversity": unique_func_calls / (num_func_calls + 1),
        "ast_calls_per_func": num_func_calls / safe_funcs,
        "ast_struct_defs": struct_defs, "ast_enum_defs": enum_defs,
        "ast_typedef_count": typedef_count,
        "ast_pointer_decls": pointer_decls, "ast_arrow_ops": arrow_ops,
        "ast_deref_count": deref_count, "ast_array_decls": array_decls,
        "ast_define_count": defines, "ast_ifdef_count": ifdefs,
        "ast_pragma_count": pragmas, "ast_return_count": return_count,
        "ast_returns_per_func": return_count / safe_funcs,
        "ast_avg_func_length": avg_func_len,
        "ast_std_func_length": std_func_len,
        "ast_max_func_length": max_func_len,
        "ast_avg_params": avg_params,
        "ast_max_paren_depth": max_paren_depth,
    }

def extract_style_features(code):
    lines = code.split("\n")
    num_lines = len(lines)
    line_lengths = [len(l) for l in lines]
    leading_spaces = [len(l) - len(l.lstrip(" ")) for l in lines]
    total_spaces = code.count(" ")
    total_tabs = code.count("\t")
    include_lines = [l for l in lines if re.match(r"\s*#\s*include", l)]
    import_count = len(include_lines)
    single_comments = sum(1 for l in lines if re.search(r"//", l))
    block_open = code.count("/*")
    open_braces = code.count("{")
    open_parens = code.count("(")
    semicolons = code.count(";")
    camel = len(re.findall(r"[a-z][A-Z]", code))
    snake = len(re.findall(r"[a-z]_[a-z]", code))
    blank_lines = sum(1 for l in lines if not l.strip())
    consecutive_blanks = len(re.findall(r"\n\n\n", code))
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
    string_lits = len(re.findall(r'"[^"]*"', code))
    unique_ratio = len(set(word_tokens)) / total_tokens
    safe_lines = num_lines or 1
    feats = {
        "num_lines": num_lines, "num_chars": len(code),
        "entropy": shannon_entropy(code),
        "mean_line_len": np.mean(line_lengths),
        "median_line_len": np.median(line_lengths),
        "std_line_len": np.std(line_lengths),
        "max_line_len": max(line_lengths) if line_lengths else 0,
        "min_line_len": min(line_lengths) if line_lengths else 0,
        "mean_indent": np.mean(leading_spaces),
        "std_indent": np.std(leading_spaces),
        "max_indent": max(leading_spaces) if leading_spaces else 0,
        "tab_count": total_tabs, "space_count": total_spaces,
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
        "camel_case_count": camel, "snake_case_count": snake,
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

# ── 4. Process all samples ──────────────────────────────────────────

print("Extracting stylometric + AST features …")
source_codes = []
rows = []
labels = []
for model, entry in sampled:
    code = entry.get("source_code", "") or ""
    code = strip_model_identifiers(code)
    source_codes.append(code)
    feats = extract_style_features(code)
    feats.update(extract_ast_features(code))
    rows.append(feats)
    labels.append(model)

df = pd.DataFrame(rows)
df["model"] = labels
print(f"Structured features: {df.shape[1] - 1} columns")

# ── 5. Prepare splits (before CodeBERT, so we know train/test) ──────

le = LabelEncoder()
y = le.fit_transform(df["model"])
X_structured = df.drop(columns=["model"]).values

X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    np.arange(len(y)), y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

X_struct_train = X_structured[X_train_idx]
X_struct_test = X_structured[X_test_idx]
codes_train = [source_codes[i] for i in X_train_idx]
codes_test = [source_codes[i] for i in X_test_idx]

# ── 6. CodeBERT embeddings (1000/model subsample for training) ──────

BERT_SAMPLE = 1000

if os.path.exists(EMB_CACHE_TRAIN) and os.path.exists(EMB_CACHE_TEST) and os.path.exists(EMB_CACHE_LABELS):
    print("\nLoading cached CodeBERT embeddings …")
    emb_train = np.load(EMB_CACHE_TRAIN)
    emb_test = np.load(EMB_CACHE_TEST)
    bert_train_labels = np.load(EMB_CACHE_LABELS)
    print(f"CodeBERT embeddings: train={emb_train.shape}, test={emb_test.shape}")
else:
    print(f"\nSubsampling {BERT_SAMPLE}/model for CodeBERT training …")

    bert_train_idx = []
    for cls_id in range(len(le.classes_)):
        cls_mask = np.where(y_train == cls_id)[0]
        chosen = rng.choice(cls_mask, size=min(BERT_SAMPLE, len(cls_mask)), replace=False)
        bert_train_idx.extend(chosen)
    bert_train_idx = np.array(bert_train_idx)
    rng.shuffle(bert_train_idx)

    bert_train_codes = [codes_train[i] for i in bert_train_idx]
    bert_train_labels = y_train[bert_train_idx]

    print(f"Loading CodeBERT (microsoft/codebert-base) on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = AutoModel.from_pretrained("microsoft/codebert-base")
    codebert = codebert.to(DEVICE)
    codebert.eval()

    def encode_batch(code_list):
        all_emb = []
        for bs in range(0, len(code_list), BATCH_SIZE):
            be = min(bs + BATCH_SIZE, len(code_list))
            encoded = tokenizer(
                code_list[bs:be], padding=True, truncation=True,
                max_length=MAX_TOKEN_LEN, return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                out = codebert(**encoded)
                emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(emb)
            if (bs // BATCH_SIZE) % 50 == 0:
                print(f"  {bs}/{len(code_list)} …", flush=True)
        return np.vstack(all_emb)

    print(f"Encoding {len(bert_train_codes)} training samples …", flush=True)
    emb_train = encode_batch(bert_train_codes)
    print(f"Encoding {len(codes_test)} test samples …", flush=True)
    emb_test = encode_batch(codes_test)
    print(f"CodeBERT embeddings: train={emb_train.shape}, test={emb_test.shape}")

    np.save(EMB_CACHE_TRAIN, emb_train)
    np.save(EMB_CACHE_TEST, emb_test)
    np.save(EMB_CACHE_LABELS, bert_train_labels)
    print("Saved embedding caches to disk.")

    del codebert, tokenizer
    import gc; gc.collect()

# Free heavy objects before training phase
del sampled, source_codes, rows
gc.collect()

# ── 7. TF-IDF text model ────────────────────────────────────────────

print("\nBuilding TF-IDF model …")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_char = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 5),
    max_features=50000, sublinear_tf=True, strip_accents="unicode",
)
tfidf_word = TfidfVectorizer(
    analyzer="word", token_pattern=r"\b\w+\b",
    ngram_range=(1, 2), max_features=30000, sublinear_tf=True,
)

X_tfidf_char_train = tfidf_char.fit_transform(codes_train)
X_tfidf_char_test = tfidf_char.transform(codes_test)
X_tfidf_word_train = tfidf_word.fit_transform(codes_train)
X_tfidf_word_test = tfidf_word.transform(codes_test)

X_tfidf_train = hstack([X_tfidf_char_train, X_tfidf_word_train])
X_tfidf_test = hstack([X_tfidf_char_test, X_tfidf_word_test])
print(f"TF-IDF shape: {X_tfidf_train.shape}")

lr_tfidf = LogisticRegression(C=5.0, max_iter=1000, solver="saga", random_state=RANDOM_STATE)
lr_tfidf.fit(X_tfidf_train, y_train)
lr_tfidf_acc = accuracy_score(y_test, lr_tfidf.predict(X_tfidf_test))
print(f"TF-IDF LR accuracy: {lr_tfidf_acc:.4f}")

# ── 8. CodeBERT LR model ────────────────────────────────────────────

print("\nTraining CodeBERT embedding classifier …")
scaler = StandardScaler()
emb_train_scaled = scaler.fit_transform(emb_train)
emb_test_scaled = scaler.transform(emb_test)

lr_bert = LogisticRegression(C=1.0, max_iter=1000, solver="saga", random_state=RANDOM_STATE)
lr_bert.fit(emb_train_scaled, bert_train_labels)
lr_bert_acc = accuracy_score(y_test, lr_bert.predict(emb_test_scaled))
print(f"CodeBERT LR accuracy: {lr_bert_acc:.4f}")

# ── 9. XGBoost (tuned from previous run) ────────────────────────────

print("\nTraining XGBoost …")
xgb_clf = xgb.XGBClassifier(
    n_estimators=389, max_depth=8, learning_rate=0.08,
    subsample=0.94, colsample_bytree=0.82, min_child_weight=4,
    gamma=0.17, reg_alpha=0.4, reg_lambda=2.28,
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
)
xgb_clf.fit(X_struct_train, y_train, eval_set=[(X_struct_test, y_test)], verbose=100)
xgb_acc = accuracy_score(y_test, xgb_clf.predict(X_struct_test))
print(f"XGBoost accuracy: {xgb_acc:.4f}")

# ── 10. 3-way ensemble: grid search weights ─────────────────────────

print("\n" + "=" * 60)
print("3-WAY ENSEMBLE (XGBoost + TF-IDF LR + CodeBERT LR)")
print("=" * 60)

xgb_proba = xgb_clf.predict_proba(X_struct_test)
tfidf_proba = lr_tfidf.predict_proba(X_tfidf_test)
bert_proba = lr_bert.predict_proba(emb_test_scaled)

best_acc = 0
best_weights = (0, 0, 0)
for w_xgb in np.arange(0.0, 1.01, 0.05):
    for w_tfidf in np.arange(0.0, 1.01 - w_xgb, 0.05):
        w_bert = round(1.0 - w_xgb - w_tfidf, 2)
        if w_bert < 0:
            continue
        combined = w_xgb * xgb_proba + w_tfidf * tfidf_proba + w_bert * bert_proba
        preds = np.argmax(combined, axis=1)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_weights = (w_xgb, w_tfidf, w_bert)

w_xgb, w_tfidf, w_bert = best_weights
print(f"Best weights: XGBoost={w_xgb:.2f}, TF-IDF={w_tfidf:.2f}, CodeBERT={w_bert:.2f}")
print(f"Ensemble accuracy: {best_acc:.4f}  ({best_acc * 100:.2f}%)\n")

combined_proba = w_xgb * xgb_proba + w_tfidf * tfidf_proba + w_bert * bert_proba
y_pred_ensemble = np.argmax(combined_proba, axis=1)

# ── 11. Full report ─────────────────────────────────────────────────

print("=" * 60)
print("FULL CLASSIFICATION REPORT — 3-WAY ENSEMBLE")
print("=" * 60)
print(classification_report(y_test, y_pred_ensemble, target_names=le.classes_, digits=3))

cm = confusion_matrix(y_test, y_pred_ensemble)
print("Confusion Matrix:")
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df.to_string())

# ── 12. Per-model comparison ────────────────────────────────────────

y_pred_xgb = xgb_clf.predict(X_struct_test)
y_pred_tfidf = lr_tfidf.predict(X_tfidf_test)
y_pred_bert = lr_bert.predict(emb_test_scaled)

print(f"\n{'=' * 70}")
print("COMPARISON: All 4 approaches")
print(f"{'=' * 70}")

rows_comp = []
for i, cls in enumerate(le.classes_):
    mask = y_test == i
    rows_comp.append({
        "Model": cls,
        "N": mask.sum(),
        "XGB": f"{accuracy_score(y_test[mask], y_pred_xgb[mask]):.3f}",
        "TFIDF": f"{accuracy_score(y_test[mask], y_pred_tfidf[mask]):.3f}",
        "CodeBERT": f"{accuracy_score(y_test[mask], y_pred_bert[mask]):.3f}",
        "Ensemble": f"{accuracy_score(y_test[mask], y_pred_ensemble[mask]):.3f}",
        "Ens_F1": f"{f1_score(y_test, y_pred_ensemble, labels=[i], average=None)[0]:.3f}",
    })

comp_df = pd.DataFrame(rows_comp)
comp_df = comp_df.sort_values("Ens_F1", ascending=False)
print(comp_df.to_string(index=False))

print(f"\n{'=' * 70}")
print("OVERALL ACCURACY SUMMARY")
print(f"{'=' * 70}")
print(f"  XGBoost (stylometric+AST):     {xgb_acc:.4f}")
print(f"  TF-IDF Logistic Regression:    {lr_tfidf_acc:.4f}")
print(f"  CodeBERT Logistic Regression:  {lr_bert_acc:.4f}")
print(f"  3-Way Ensemble:                {best_acc:.4f}")
