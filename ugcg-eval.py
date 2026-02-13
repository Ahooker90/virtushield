# Model Performance Evaluation Script
# Run this in a Jupyter notebook cell after importing your moderation code

import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import pandas as pd
from moderation_pipeline_v5 import SupervisorAgent
def evaluate_model(sexual_folder="unsafe-final", 
                   safe_folder="SAFE",
                   num_images=-1):
    """
    Evaluate model performance on labeled datasets.
    
    Args:
        sexual_folder: Path to folder with sexual/unsafe images
        safe_folder: Path to folder with safe images  
        num_images: Number of images to process per folder (-1 for all)
    """
    
    print("=" * 70)
    print("🧪 STARTING MODEL EVALUATION")
    print("=" * 70)
    
    # Initialize supervisor
    supervisor = SupervisorAgent()
    
    y_true = []  # Ground truth labels
    y_pred = []  # Model predictions
    
    # Process UNSAFE images (sexual folder)
    print(f"\n📁 Processing UNSAFE images from: {sexual_folder}")
    print("-" * 70)
    
    sexual_files = [f for f in os.listdir(sexual_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if num_images == -1:
        num_sexual = len(sexual_files)
    else:
        num_sexual = min(num_images, len(sexual_files))
    
    print(f"📊 Processing {num_sexual} images from sexual folder...")
    
    for idx, filename in enumerate(sexual_files[:num_sexual]):
        path = os.path.join(sexual_folder, filename)
        result = supervisor.plan_and_moderate(path)
        
        # Ground truth: UNSAFE (1)
        y_true.append(1)
        # Prediction: UNSAFE=1, SAFE=0
        y_pred.append(1 if result['overall'] == "UNSAFE" else 0)
        
        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx + 1}/{num_sexual} images processed")
    
    print(f"✅ Completed sexual folder: {num_sexual} images")
    
    # Process SAFE images (safe folder)
    print(f"\n📁 Processing SAFE images from: {safe_folder}")
    print("-" * 70)
    
    safe_files = [f for f in os.listdir(safe_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if num_images == -1:
        num_safe = len(safe_files)
    else:
        num_safe = min(num_images, len(safe_files))
    
    print(f"📊 Processing {num_safe} images from safe folder...")
    
    for idx, filename in enumerate(safe_files[:num_safe]):
        path = os.path.join(safe_folder, filename)
        result = supervisor.plan_and_moderate(path)
        
        # Ground truth: SAFE (0)
        y_true.append(0)
        # Prediction: UNSAFE=1, SAFE=0
        y_pred.append(1 if result['overall'] == "UNSAFE" else 0)
        
        if (idx + 1) % 10 == 0:
            print(f"   Progress: {idx + 1}/{num_safe} images processed")
    
    print(f"✅ Completed safe folder: {num_safe} images")
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("📊 CALCULATING PERFORMANCE METRICS")
    print("=" * 70)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
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
    # Evaluate on all images (use num_images parameter to limit)
    results = evaluate_model(
        sexual_folder="in-the-wild/nsfw",
        safe_folder="in-the-wild/safe",
        num_images=-1  # -1 for all images, or specify a number like 50
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