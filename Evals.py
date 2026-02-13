# Model Performance Evaluation Script (recursive folder traversal)
# Run this in a Jupyter notebook cell after importing your moderation code

import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import pandas as pd
from moderation_pipeline_v5 import SupervisorAgent

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

def list_images_recursive(root_dir):
    """Return a list of image file paths by walking all subfolders of root_dir."""
    if not os.path.isdir(root_dir):
        print(f"⚠️  Directory not found: {root_dir}")
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(dirpath, fn))
    # Optional: sort for deterministic ordering
    paths.sort()
    return paths

def evaluate_model(sexual_folder="unsafe-final", 
                   safe_folder="SAFE",
                   num_images=-1):
    """
    Evaluate model performance on labeled datasets by recursively scanning subfolders.
    
    Args:
        sexual_folder: Path to folder with sexual/unsafe images (scans subfolders)
        safe_folder: Path to folder with safe images (scans subfolders)
        num_images: Max number of images to process per class total (-1 for all)
    """
    
    print("=" * 70)
    print("🧪 STARTING MODEL EVALUATION (recursive)")
    print("=" * 70)
    
    # Initialize supervisor
    supervisor = SupervisorAgent()
    
    y_true = []  # Ground truth labels
    y_pred = []  # Model predictions
    
    # ---------------------------
    # Process UNSAFE images
    # ---------------------------
    print(f"\n📁 Processing UNSAFE images from: {sexual_folder}")
    print("-" * 70)
    sexual_files = list_images_recursive(sexual_folder)

    if num_images == -1:
        sexual_subset = sexual_files
    else:
        sexual_subset = sexual_files[:min(num_images, len(sexual_files))]
    
    print(f"📊 Found {len(sexual_files)} UNSAFE images (processing {len(sexual_subset)})...")
    
    for idx, path in enumerate(sexual_subset, start=1):
        result = supervisor.plan_and_moderate(path)
        # Ground truth: UNSAFE (1)
        y_true.append(1)
        # Prediction: UNSAFE=1, SAFE=0
        y_pred.append(1 if result['overall'] == "UNSAFE" else 0)
        
        if idx % 25 == 0:
            print(f"   Progress (UNSAFE): {idx}/{len(sexual_subset)}")
    
    print(f"✅ Completed sexual folder: {len(sexual_subset)} images")
    
    # ---------------------------
    # Process SAFE images
    # ---------------------------
    print(f"\n📁 Processing SAFE images from: {safe_folder}")
    print("-" * 70)
    safe_files = list_images_recursive(safe_folder)

    if num_images == -1:
        safe_subset = safe_files
    else:
        safe_subset = safe_files[:min(num_images, len(safe_files))]
    
    print(f"📊 Found {len(safe_files)} SAFE images (processing {len(safe_subset)})...")
    
    for idx, path in enumerate(safe_subset, start=1):
        result = supervisor.plan_and_moderate(path)
        # Ground truth: SAFE (0)
        y_true.append(0)
        # Prediction: UNSAFE=1, SAFE=0
        y_pred.append(1 if result['overall'] == "UNSAFE" else 0)
        
        if idx % 25 == 0:
            print(f"   Progress (SAFE): {idx}/{len(safe_subset)}")
    
    print(f"✅ Completed safe folder: {len(safe_subset)} images")
    
    # ---------------------------
    # Calculate metrics
    # ---------------------------
    print("\n" + "=" * 70)
    print("📊 CALCULATING PERFORMANCE METRICS")
    print("=" * 70)

    # Guard against edge cases where one class might be absent in y_true
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix with fixed labels => always 2x2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Display results
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n🔢 CONFUSION MATRIX:")
    print(f"   True Negatives (TN):  {tn} (Correctly predicted SAFE)")
    print(f"   False Positives (FP): {fp} (SAFE predicted as UNSAFE)")
    print(f"   False Negatives (FN): {fn} (UNSAFE predicted as SAFE)")
    print(f"   True Positives (TP):  {tp} (Correctly predicted UNSAFE)")
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total images: {len(y_true)}")
    print(f"   Actual UNSAFE: {sum(y_true)}")
    print(f"   Actual SAFE: {len(y_true) - sum(y_true)}")
    print(f"   Predicted UNSAFE: {sum(y_pred)}")
    print(f"   Predicted SAFE: {len(y_pred) - sum(y_pred)}")
    
    print("\n" + "=" * 70)
    
    # Return results as dictionary
    results = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    return results

# Run evaluation
if __name__ == "__main__":
    # Evaluate on all images (use num_images parameter to limit the total per class)
    results = evaluate_model(
        sexual_folder="final-datasets/final-filtered-data/unsafe",
        safe_folder="final-datasets/final-filtered-data/safe",
        num_images=-1  # -1 for all images, or specify a number like 500
    )
    
    # Optional: Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'Accuracy', 'F1-Score'],
        'Value': [
            results['precision'],
            results['recall'],
            results['accuracy'],
            results['f1_score']
        ]
    })
    
    print("\n📋 SUMMARY TABLE:")
    print(summary_df.to_string(index=False))
