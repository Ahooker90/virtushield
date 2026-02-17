"""
Ablation Study Runner: Full 2x2 Factorial (Context Encoding x Reflection) across all datasets.

Runs 4 configurations on 3 datasets (12 experimental conditions total).
Produces per-image CSVs, per-dataset summaries, and aggregate statistics.
Includes checkpointing for long runs and McNemar's statistical tests.
"""

import os
import sys
import json
import time
import csv
import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

# Ensure working directory and .env are set before pipeline import
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
for env_path in [os.path.join(PROJECT_ROOT, ".env"), "/Users/ahooker/Documents/Dissertation/virtushield/.env"]:
    if os.path.isfile(env_path):
        load_dotenv(env_path, override=True)
        break

# Import pipeline components
from moderation_pipeline_v5 import SupervisorAgent

# ─── Configuration Matrix (2x2 factorial) ─────────────────────────────────────

ABLATION_CONFIGS = {
    "baseline":              {"activate_yolo": False, "context_encoding": False, "reflection_active": False},
    "context_encoding_only": {"activate_yolo": False, "context_encoding": True,  "reflection_active": False},
    "reflection_only":       {"activate_yolo": True,  "context_encoding": False, "reflection_active": True},
    "full_system":           {"activate_yolo": True,  "context_encoding": True,  "reflection_active": True},
}

# ─── Dataset Registry ─────────────────────────────────────────────────────────

BASE_DIR = "/Users/ahooker/Documents/Dissertation/virtushield"

DATASETS = {
    "vrchat": {
        "unsafe_dir": f"{BASE_DIR}/final-datasets/final-filtered-data/unsafe",
        "safe_dir": f"{BASE_DIR}/final-datasets/final-filtered-data/safe",
        "description": "VRChat controlled dataset",
    },
    "second_life": {
        "unsafe_dir": f"{BASE_DIR}/final-datasets/sl-dataset/unsafe-final",
        "safe_dir": f"{BASE_DIR}/final-datasets/sl-dataset/safe-final",
        "description": "Second Life in-the-wild",
    },
    "in_the_wild": {
        "unsafe_dir": f"{BASE_DIR}/final-datasets/in-the-wild-final-small/UNSAFE",
        "safe_dir": f"{BASE_DIR}/final-datasets/in-the-wild-final-small/SAFE",
        "description": "Streaming video frames in-the-wild",
    },
}

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
OUTPUT_DIR = "ablation_results"


def list_images_recursive(root_dir):
    """Return sorted list of image file paths by walking all subfolders."""
    if not os.path.isdir(root_dir):
        print(f"  WARNING: Directory not found: {root_dir}")
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def compute_metrics(y_true, y_pred):
    """Compute classification metrics. Returns dict."""
    if len(y_true) == 0:
        return {"precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": precision, "recall": recall,
        "accuracy": accuracy, "f1": f1,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": len(y_true),
    }


def evaluate_config_on_dataset(config, config_name, dataset_name, dataset_info, output_dir):
    """Run one config on one dataset. Returns metrics dict with y_true/y_pred."""
    print(f"\n{'='*70}")
    print(f"  CONFIG: {config_name} | DATASET: {dataset_name}")
    print(f"  CE={config['context_encoding']}, Reflection={config['reflection_active']}")
    print(f"{'='*70}")

    unsafe_dir = dataset_info["unsafe_dir"]
    safe_dir = dataset_info["safe_dir"]

    unsafe_files = list_images_recursive(unsafe_dir)
    safe_files = list_images_recursive(safe_dir)

    print(f"  Found {len(unsafe_files)} UNSAFE, {len(safe_files)} SAFE images")

    supervisor = SupervisorAgent(config=config)

    y_true = []
    y_pred = []
    per_image_rows = []

    # Process UNSAFE images
    for idx, path in enumerate(unsafe_files, 1):
        try:
            result = supervisor.plan_and_moderate(path)
            pred = 1 if result["overall"] == "UNSAFE" else 0
            prob = max((r["nsfw_prob"] for r in result["regions"]), default=0.0)
        except Exception as e:
            print(f"  ERROR processing {path}: {e}")
            pred = 0
            prob = 0.0
        y_true.append(1)
        y_pred.append(pred)
        per_image_rows.append({
            "image_path": path, "ground_truth": "UNSAFE", "prediction": "UNSAFE" if pred == 1 else "SAFE",
            "nsfw_prob": prob, "config": config_name, "dataset": dataset_name,
        })
        if idx % 50 == 0:
            print(f"  Progress (UNSAFE): {idx}/{len(unsafe_files)}")

    # Process SAFE images
    for idx, path in enumerate(safe_files, 1):
        try:
            result = supervisor.plan_and_moderate(path)
            pred = 1 if result["overall"] == "UNSAFE" else 0
            prob = max((r["nsfw_prob"] for r in result["regions"]), default=0.0)
        except Exception as e:
            print(f"  ERROR processing {path}: {e}")
            pred = 0
            prob = 0.0
        y_true.append(0)
        y_pred.append(pred)
        per_image_rows.append({
            "image_path": path, "ground_truth": "SAFE", "prediction": "UNSAFE" if pred == 1 else "SAFE",
            "nsfw_prob": prob, "config": config_name, "dataset": dataset_name,
        })
        if idx % 50 == 0:
            print(f"  Progress (SAFE): {idx}/{len(safe_files)}")

    # Save per-image CSV
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, f"per_image_{config_name}_{dataset_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth", "prediction", "nsfw_prob", "config", "dataset"])
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"  Saved per-image results to {csv_path}")

    metrics = compute_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["config"] = config_name
    metrics["dataset"] = dataset_name

    print(f"  RESULTS: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f}")
    return metrics


def save_checkpoint(all_results, output_dir):
    """Save checkpoint (without y_true/y_pred arrays for JSON compatibility)."""
    checkpoint = {}
    for key, result in all_results.items():
        config_name, dataset_name = key
        checkpoint[f"{config_name}__{dataset_name}"] = {
            k: v for k, v in result.items() if k not in ("y_true", "y_pred")
        }
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(output_dir):
    """Load completed (config, dataset) pairs from checkpoint."""
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path) as f:
        data = json.load(f)
    return set(k for k in data.keys())


def load_per_image_csv(output_dir, config_name, dataset_name):
    """Reload y_true/y_pred from a previously saved per-image CSV."""
    csv_path = os.path.join(output_dir, "raw", f"per_image_{config_name}_{dataset_name}.csv")
    if not os.path.exists(csv_path):
        return None
    y_true, y_pred = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(1 if row["ground_truth"] == "UNSAFE" else 0)
            y_pred.append(1 if row["prediction"] == "UNSAFE" else 0)
    metrics = compute_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["config"] = config_name
    metrics["dataset"] = dataset_name
    return metrics


def compute_aggregate(all_results):
    """Pool y_true/y_pred across all datasets for each config. Returns dict keyed by config_name."""
    aggregate = {}
    for config_name in ABLATION_CONFIGS:
        pooled_true, pooled_pred = [], []
        for dataset_name in DATASETS:
            key = (config_name, dataset_name)
            if key in all_results:
                pooled_true.extend(all_results[key]["y_true"])
                pooled_pred.extend(all_results[key]["y_pred"])
        if pooled_true:
            metrics = compute_metrics(pooled_true, pooled_pred)
            metrics["y_true"] = pooled_true
            metrics["y_pred"] = pooled_pred
            metrics["config"] = config_name
            metrics["dataset"] = "aggregate"
            aggregate[config_name] = metrics
    return aggregate


def save_summary_csvs(all_results, aggregate, output_dir):
    """Save per-dataset and aggregate summary CSVs."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Per-dataset CSV
    per_dataset_path = os.path.join(tables_dir, "ablation_per_dataset.csv")
    with open(per_dataset_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "dataset", "precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn", "n"])
        writer.writeheader()
        for (config_name, dataset_name), result in sorted(all_results.items()):
            writer.writerow({
                "config": config_name, "dataset": dataset_name,
                "precision": f"{result['precision']:.4f}", "recall": f"{result['recall']:.4f}",
                "f1": f"{result['f1']:.4f}", "accuracy": f"{result['accuracy']:.4f}",
                "tp": result["tp"], "fp": result["fp"], "tn": result["tn"], "fn": result["fn"],
                "n": result["n"],
            })
    print(f"\nSaved per-dataset summary to {per_dataset_path}")

    # Aggregate CSV
    agg_path = os.path.join(tables_dir, "ablation_aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn", "n"])
        writer.writeheader()
        for config_name in ABLATION_CONFIGS:
            if config_name in aggregate:
                r = aggregate[config_name]
                writer.writerow({
                    "config": config_name,
                    "precision": f"{r['precision']:.4f}", "recall": f"{r['recall']:.4f}",
                    "f1": f"{r['f1']:.4f}", "accuracy": f"{r['accuracy']:.4f}",
                    "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
                    "n": r["n"],
                })
    print(f"Saved aggregate summary to {agg_path}")


def run_full_ablation(output_dir=OUTPUT_DIR):
    """Main entry point: run all 4 configs x 3 datasets with checkpointing."""
    os.makedirs(output_dir, exist_ok=True)
    completed = load_checkpoint(output_dir)

    print(f"\n{'#'*70}")
    print(f"  ABLATION STUDY: 2x2 Factorial (CE x Reflection) x 3 Datasets")
    print(f"  Output directory: {output_dir}")
    print(f"  Already completed: {len(completed)}/12 conditions")
    print(f"  Started: {datetime.datetime.now().isoformat()}")
    print(f"{'#'*70}")

    all_results = {}

    # Reload previously completed results from CSVs
    for key_str in completed:
        config_name, dataset_name = key_str.split("__")
        cached = load_per_image_csv(output_dir, config_name, dataset_name)
        if cached:
            all_results[(config_name, dataset_name)] = cached
            print(f"  Loaded cached: {config_name} x {dataset_name} (F1={cached['f1']:.4f})")

    # Run remaining conditions (smallest dataset first for fast feedback)
    dataset_order = ["second_life", "in_the_wild", "vrchat"]

    for dataset_name in dataset_order:
        for config_name in ABLATION_CONFIGS:
            key_str = f"{config_name}__{dataset_name}"
            if key_str in completed:
                continue

            start_time = time.time()
            result = evaluate_config_on_dataset(
                ABLATION_CONFIGS[config_name], config_name,
                dataset_name, DATASETS[dataset_name], output_dir
            )
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed/60:.1f} minutes")

            all_results[(config_name, dataset_name)] = result
            save_checkpoint(all_results, output_dir)

    # Compute aggregate statistics
    aggregate = compute_aggregate(all_results)

    # Save summary tables
    save_summary_csvs(all_results, aggregate, output_dir)

    # Run statistical analysis
    print(f"\n{'='*70}")
    print("  RUNNING STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    try:
        from ablation_statistics import run_all_statistics
        run_all_statistics(all_results, aggregate, output_dir)
    except ImportError as e:
        print(f"  WARNING: Could not import ablation_statistics: {e}")
        print("  Skipping statistical analysis.")

    # Generate visualizations
    print(f"\n{'='*70}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    try:
        from ablation_visualizations import generate_all_figures
        generate_all_figures(output_dir)
    except ImportError as e:
        print(f"  WARNING: Could not import ablation_visualizations: {e}")
        print("  Skipping visualizations.")

    print(f"\n{'#'*70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Finished: {datetime.datetime.now().isoformat()}")
    print(f"{'#'*70}")

    # Print summary table
    print(f"\n{'='*70}")
    print("  SUMMARY: F1 Scores")
    print(f"{'='*70}")
    header = f"  {'Config':<25}"
    for ds in dataset_order:
        header += f" {ds:<15}"
    header += f" {'aggregate':<15}"
    print(header)
    print("  " + "-" * 85)
    for config_name in ["baseline", "context_encoding_only", "reflection_only", "full_system"]:
        row = f"  {config_name:<25}"
        for ds in dataset_order:
            key = (config_name, ds)
            if key in all_results:
                row += f" {all_results[key]['f1']:<15.4f}"
            else:
                row += f" {'N/A':<15}"
        if config_name in aggregate:
            row += f" {aggregate[config_name]['f1']:<15.4f}"
        print(row)

    return all_results, aggregate


if __name__ == "__main__":
    run_full_ablation()
