"""Occurrence-aware, article-stratified evaluation; gold never enters extraction."""
from collections import defaultdict
from difflib import SequenceMatcher
import re


def canonical(text, layout_only=False):
    return re.sub(r"\s+", "", text) if layout_only else text


def evaluate(predictions: list[dict], gold: list[dict], *, layout_only=False) -> dict:
    pools = defaultdict(list)
    for index, q in enumerate(gold):
        pools[(str(q["article_id"]), canonical(q["text"], layout_only))].append(index)
    matched, extras, source_added, source_wrong, source_correct = set(), [], [], [], 0
    label_pairs, dimension_tp, dimension_fp, dimension_fn = [], 0, 0, 0
    for p in predictions:
        key = (str(p["article_id"]), canonical(p["text"], layout_only))
        remaining = [i for i in pools[key] if i not in matched]
        if not remaining:
            extras.append(p)
            continue
        index = remaining[0]
        matched.add(index)
        ps, gs = p.get("source_explicit", {}), gold[index].get("source_explicit", {})
        # A missing gold source field means unlabelled, not a verified negative.
        if "source_explicit" in gold[index]:
            if "book" in gs and not gs.get("book") and ps.get("book"):
                source_added.append(p.get("id", key[0]))
            elif any(ps.get(field) != gs[field] for field in ("book", "chapter") if field in gs):
                source_wrong.append(p.get("id", key[0]))
            else:
                source_correct += 1
        gr, pr = gold[index].get("relevance", {}), p.get("relevance", {})
        if gr.get("label"):
            label_pairs.append((gr["label"], pr.get("label", "unclassified")))
        if "dimensions" in gr:
            gd, pd = set(gr["dimensions"]), set(pr.get("dimensions", []))
            dimension_tp += len(gd & pd); dimension_fp += len(pd - gd); dimension_fn += len(gd - pd)
    missing = [g for i, g in enumerate(gold) if i not in matched]
    tp, fp, fn = len(matched), len(extras), len(missing)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    near = []
    for g in missing:
        candidates = [p for p in extras if str(p["article_id"]) == str(g["article_id"])]
        if candidates:
            best = max(candidates, key=lambda p: SequenceMatcher(None, g["text"], p["text"]).ratio())
            score = SequenceMatcher(None, g["text"], best["text"]).ratio()
            if score >= 0.5:
                near.append({"gold": g["text"], "prediction": best["text"], "similarity": round(score, 4),
                             "article_id": g["article_id"], "counts_as_match": False})
    confusion = defaultdict(int)
    for truth, prediction in label_pairs:
        confusion[f"{truth} -> {prediction}"] += 1
    dim_den = 2 * dimension_tp + dimension_fp + dimension_fn
    return {"mode": "whitespace_insensitive" if layout_only else "exact", "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            **({"precision": None, "recall": None, "f1": None} if not gold else {}),
            "metrics_available": bool(gold), "gold_count": len(gold), "prediction_count": len(predictions),
            "classification_on_matched_quotes": {"labelled_pairs": len(label_pairs), "confusion": dict(confusion),
                "accuracy": sum(g == p for g, p in label_pairs) / len(label_pairs) if label_pairs else None,
                "dimensions_micro_f1": 2 * dimension_tp / dim_den if dim_den else None},
            "duplicate_alignment": "occurrence order within article and exact quote text; keep gold in source order",
            "source_false_additions": source_added, "source_mismatches": source_wrong,
            "source_exact_correct": source_correct, "missing": missing, "extra": extras,
            "near_matches_diagnostic_only": near}
