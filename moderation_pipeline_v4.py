import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import logging
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv
import time
reflection_active = True  #Toggle for Reflection ablation study
activate_yolo = True  # Toggle for Nudity Detection ablation study
# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
OPENAI_API = os.getenv('OPENAI_API')
openai.api_key = OPENAI_API
client = OpenAI(api_key=openai.api_key)
print("🚀 INITIALIZING MULTI-AGENT VR CONTENT MODERATION SYSTEM")
print("=" * 60)
print(f"🔬 YOLO NSFW labels: {'ENABLED' if activate_yolo else 'DISABLED (Ablation Study)'}")
yolo_model = YOLO(r"yolo_ft.pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Loading meta LLM...")
print("✅ Using OpenAI API for meta-prompting")
print("✅ Models loaded successfully!")
print("=" * 60)

# Potential avatar classes from COCO (0: person, 15-22: animals like cat, dog, etc.)
avatar_classes = [0, 15, 16, 17, 18, 19, 20, 21, 22]  # Expanded for nonhuman avatars

base_prompt = "A VR scene featuring characters or avatars, or with no characters present."
explicit_prompt = "showing explicit nudity, including naked breasts, genitals, buttocks, penis, butt cheeks, or suggestive scenes without characters."
safe_prompt = "showing no explicit nudity, with characters fully or partially clothed, or a scene without nudity and no characters present."
nsfw_vs_safe_variants = [
    [f"{base_prompt} {explicit_prompt}", f"{base_prompt} {safe_prompt}"],
]

def take_consensus(probs):
    """Average probabilities across variants; decide based on higher average."""
    avg_probs = np.mean(probs, axis=0)
    print(f"    📊 Consensus probabilities: {avg_probs}")
    print(f"    🎯 Winner: Option {np.argmax(avg_probs)} with {avg_probs[np.argmax(avg_probs)]:.3f} confidence")
    return np.argmax(avg_probs)

def parse_refined_prompts(refined_output):
    """Simple parser for the output."""
    try:
        # Handle case where the output starts directly with the unsafe content (no "Unsafe: " prefix)
        # since the prompt ends with "Unsafe:"
        
        # First, check if "Unsafe: " is in the outputs
        if "Unsafe: " in refined_output:
            unsafe_part = refined_output.split("Unsafe: ")[1]
        else:
            # If not, the output likely starts with the unsafe content directly
            unsafe_part = refined_output
        
        # Split by "Safe:" to get both parts - handle both " Safe: " and "\nSafe: "
        if " Safe: " in unsafe_part:
            unsafe_content = unsafe_part.split(" Safe: ")[0]
            safe_content = unsafe_part.split(" Safe: ")[1]
        elif "\nSafe: " in unsafe_part:
            unsafe_content = unsafe_part.split("\nSafe: ")[0]
            safe_content = unsafe_part.split("\nSafe: ")[1]
        elif "Safe: " in unsafe_part:
            unsafe_content = unsafe_part.split("Safe: ")[0]
            safe_content = unsafe_part.split("Safe: ")[1]
        else:
            raise ValueError("Could not find 'Safe:' delimiter")
        
        return unsafe_content.strip(), safe_content.strip()
    except:
        # Fallback to defaults if parsing fails
        logging.warning("Failed to parse refined prompts; using defaults.")
        return "A VR scene with explicit NSFW content", "A VR scene with safe content"

def create_meta_prompt(nsfw_prob, current_prompts, image_features):
    """Create a meta prompt with context."""
    
    current_unsafe = current_prompts[0]
    current_safe = current_prompts[1]
    
    meta_prompt = f"""Refine NSFW detection prompts for VR avatar/content classification.

        Context:
        - Current NSFW probability: {nsfw_prob:.2f} (borderline case needing clarification)
        - Detected features: {', '.join(image_features)}
        - Previous unsafe prompt: "{current_unsafe}"
        - Previous safe prompt: "{current_safe}"
        - Goal: Create more specific prompts to resolve ambiguity

        Task: Generate TWO specific prompts that better distinguish between NSFW and safe content.
        - Focus on anatomical details and explicit content markers
        - Make prompts more precise than previous attempts
        - Each prompt should be 10-15 words maximum

        Output format EXACTLY as:
        Unsafe: [specific anatomical/explicit features]
        Safe: [specific clothing/coverage indicators]

        Example:
        Unsafe: naked human breasts genitals buttocks exposed skin
        Safe: clothed figure covered torso appropriate attire

        Now generate:
        Unsafe:
    """
    
    return meta_prompt

class DetectionAgent:
    def act(self, frame):
        """Detect potential avatar regions using YOLO."""
        print("\n🔍 DETECTION AGENT ACTIVATED")
        print("   └─ Running YOLO object detection...")
        frame = frame[..., ::-1]  # Convert RGB to BGR
        results = yolo_model(frame)
        detections = results[0].boxes
        crops = []
        boxes = []
        nsfw_flags = []  # Will contain detected NSFW labels
        nsfw_list_label = ["exposed_breast", "exposed_genitalia_f", "exposed_butt", "exposed_penis"]
        print(f"   └─ Found {len(detections)} total objects")
        
        # First, check if any NSFW labels are detected in the frame (only if activate_yolo is True)
        detected_nsfw_labels = []
        if activate_yolo:
            for detection in detections:
                class_id = int(detection.cls[0].item())
                class_name = yolo_model.names[class_id]
                if class_name in nsfw_list_label:
                    detected_nsfw_labels.append(class_name)
                    print(f"   └─ Found NSFW content: {class_name}")
        
        # Then process avatar detections
        avatar_count = 0
        for detection in detections:
            class_id = int(detection.cls[0].item())
            class_name = yolo_model.names[class_id]
            
            # Check if this is an avatar class OR (if activate_yolo) an NSFW class we want to crop
            should_crop = class_id in avatar_classes
            if activate_yolo and class_name in nsfw_list_label:
                should_crop = True
            
            if should_crop:
                avatar_count += 1
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                cropped = frame[y1:y2, x1:x2]
                cropped_pil = Image.fromarray(cropped)
                crops.append(cropped_pil)
                boxes.append((x1, y1, x2, y2))
                
                # Add the label if it's NSFW and activate_yolo is True, otherwise None
                if activate_yolo and class_name in nsfw_list_label:
                    nsfw_flags.append(class_name)
                    print(f"   └─ Object #{avatar_count} detected at [{x1}, {y1}, {x2}, {y2}] (NSFW: {class_name})")
                else:
                    nsfw_flags.append(None)
                    print(f"   └─ Object #{avatar_count} detected at [{x1}, {y1}, {x2}, {y2}] (Class: {class_name})")
        
        print(f"   └─ ✅ Detection complete: {avatar_count} objects processed")
        if activate_yolo:
            print(f"   └─ NSFW labels found in frame: {detected_nsfw_labels if detected_nsfw_labels else 'None'}")
        else:
            print(f"   └─ NSFW label detection disabled (ablation mode)")
        return crops, boxes, nsfw_flags

class NSFWMAssessmentAgent:
    def act(self, image, nsfw_flag=False):
        """Classify as UNSAFE (0) or SAFE (1) with consensus and reflection. Incorporate YOLO NSFW flag."""
        print(f"\n🛡️ NSFW ASSESSMENT AGENT ACTIVATED")
        print(f"   └─ Using unified safety criteria...")
        
        # Start with original variants
        variants = nsfw_vs_safe_variants.copy()
        
        # Incorporate YOLO NSFW labels into prompts if flagged (only if activate_yolo is True)
        if activate_yolo and nsfw_flag:
            print("   └─ YOLO NSFW detection flagged; enhancing unsafe prompts...")
            for i in range(len(variants)):
                variants[i] = [
                    variants[i][0] + f" The following has been detected: {nsfw_flag}",
                    variants[i][1]
                ]
        
        # Initial assessment with original variants
        probs = []
        print("   └─ Running safety assessments...")
        for i, prompts in enumerate(variants):
            inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            prob = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
            probs.append(prob)
            print(f"      └─ Test {i+1}: UNSAFE: {prob[0]:.3f} | SAFE: {prob[1]:.3f}")
        
        avg_probs = np.mean(probs, axis=0)
        nsfw_prob = avg_probs[0]  # Probability of NSFW
        
        print(f"   └─ Initial assessment: NSFW probability = {nsfw_prob:.3f}")
        
        # Iterative Prompt Rewriting with Meta-LLM
        max_iterations = 3 #Depth for reflection
        iteration = 0
        if reflection_active:
            while 0.4 < nsfw_prob < 0.7 and iteration < max_iterations:
                print(f"   └─ ⚠️ BORDERLINE CASE: Iter {iteration+1}/{max_iterations}. Refining prompts with meta-LLM...")
                logging.info("Engaging meta-LLM for prompt refinement.")
                
                # Enhanced Meta-LLM system prompt with more context
                current_unsafe = variants[-1][0]  # Use most recent variant
                current_safe = variants[-1][1]
                
                # Collect additional context
                image_features = []
                if activate_yolo and nsfw_flag:
                    image_features.append("YOLO-detected nudity markers")
                    print(f"   └─ 🍆 YOLO detected: {yolo_model.names[1]} and {yolo_model.names[2]}")
                image_features.extend(["visible form", "potential anatomy", "content markers"])
                
                # Create meta prompt
                meta_prompt = create_meta_prompt(nsfw_prob, [current_unsafe, current_safe], image_features)
                
                # Call OpenAI GPT-4o
                try:
                    refined_output = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"You are a prompt refinement assistant. Current assessment leans {'UNSAFE' if nsfw_prob > 0.5 else 'SAFE'} ({nsfw_prob:.2f}). Refine prompts to increase certainty in this direction. UNSAFE is anything with explicit nudity. Revealing clothes do not rise to the definition of UNSAFE"},
                            {"role": "user", "content": meta_prompt}
                        ],
                        max_tokens=60,
                        temperature=0.9,
                    ).choices[0].message.content
                    
                    # Parse output
                    new_unsafe, new_safe = parse_refined_prompts(refined_output)
                    
                    # ADD new variant instead of replacing
                    base_prompt = "A VR scene featuring characters or avatars, or with no characters present."
                    new_variant = [f"{base_prompt} {new_unsafe}", f"{base_prompt} {new_safe}"]
                    print(f"NEW VAR: {new_variant}")
                    variants.append(new_variant)
                    
                    # Test ONLY the new variant
                    inputs = clip_processor(text=new_variant, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                    outputs = clip_model(**inputs)
                    new_prob = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
                    probs.append(new_prob)
                    print(f"      └─ Refined variant {len(variants)}: UNSAFE: {new_prob[0]:.3f} | SAFE: {new_prob[1]:.3f}")
                    
                    # Recalculate consensus with ALL variants
                    print("   └─ Building consensus from all variants...")
                    index = take_consensus(probs)
                    nsfw_prob = np.mean(probs, axis=0)[0]
                    print(f"   └─ Post-refinement NSFW probability: {nsfw_prob:.3f}")
                    
                except Exception as e:
                    logging.error(f"OpenAI API error: {e}")
                    print(f"   └─ ❌ OpenAI API error, using current assessment")
                    break    
                iteration += 1
        
        # Final consensus decision
        print("   └─ Building final consensus from all variants...")
        final_classification_idx = take_consensus(probs)
        classification = "UNSAFE" if final_classification_idx == 0 else "SAFE"
        final_confidence = np.mean(probs, axis=0)
        print(f"   └─ ✅ Final verdict: {classification} (confidence: {max(final_confidence):.3f})")
        return classification, nsfw_prob

class SupervisorAgent:
    def __init__(self):
        print("\n👔 SUPERVISOR AGENT INITIALIZED")
        print("   └─ Assembling agent team...")
        self.detection_agent = DetectionAgent()
        print("   └─ ✅ Detection Agent ready")
        self.nsfw_agent = NSFWMAssessmentAgent()
        print("   └─ ✅ NSFW Assessment Agent ready")
        print("   └─ 🎯 Multi-agent system ready for moderation!")
    
    def plan_and_moderate(self, image_path):
        """Plan workflow and moderate a single image."""
        print(f"\n{'='*60}")
        print(f"📸 PROCESSING: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        image = Image.open(image_path)
        frame = np.array(image.convert("RGB"))
        
        # Step 1: Detection - Updated to receive nsfw_flags
        print("\n🎬 SUPERVISOR: Initiating detection phase...")
        crops, boxes, nsfw_flags = self.detection_agent.act(frame)
        
        results = {
            "image_path": image_path,
            "regions": [],
            "overall": "SAFE"
        }
        
        if not crops:
            print("\n🎬 SUPERVISOR: No avatars detected. Marking as SAFE without moderation.")
            results["regions"].append({
                "box": "whole_image",
                "nsfw_label": "SAFE",
                "nsfw_prob": 0.0
            })
        else:
            # Process regions - Pass nsfw_flag to NSFW agent
            print(f"\n🎬 SUPERVISOR: Processing {len(crops)} detected regions...")
            for i, crop in enumerate(crops):
                print(f"\n--- Region {i+1}/{len(crops)} ---")
                
                # NSFW assessment - Pass nsfw_flags[i]
                print(f"🎬 SUPERVISOR: Requesting NSFW assessment...")
                nsfw_label, nsfw_prob = self.nsfw_agent.act(crop, nsfw_flags[i])
                
                results["regions"].append({
                    "box": boxes[i],
                    "nsfw_label": nsfw_label,
                    "nsfw_prob": float(nsfw_prob)
                })
                
                print(f"🎬 SUPERVISOR: Region {i+1} complete - {nsfw_label}")
        
        # Aggregate: If any UNSAFE, overall UNSAFE
        print("\n🎬 SUPERVISOR: Aggregating results...")
        unsafe_count = sum(1 for r in results["regions"] if r["nsfw_label"] == "UNSAFE")
        if unsafe_count > 0:
            results["overall"] = "UNSAFE"
            print(f"   └─ ⚠️  Found {unsafe_count} UNSAFE region(s) - marking overall as UNSAFE")
        else:
            print(f"   └─ ✅ All {len(results['regions'])} region(s) are SAFE")
        
        print(f"\n📋 FINAL VERDICT: {results['overall']}")
        print(f"{'='*60}")
        
        return results

#Main Loop
def moderate_images(directory, num_images=-1):
    """Batch process images in a directory for ML evaluation."""
    print("\n" + "🎯 " * 20)
    print("STARTING BATCH MODERATION")
    print("🎯 " * 20)
    
    supervisor = SupervisorAgent()
    outputs = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(image_files)

    if num_images == -1 or num_images > total_files:
        num_images = total_files

    num_safe = 0

    print(f"\n📁 Directory: {directory}")
    print(f"📊 Total images to process: {num_images}")
    print(f"🔬 YOLO NSFW labels: {'ENABLED' if activate_yolo else 'DISABLED (Ablation Study)'}")
    print("\n" + "-" * 60)

    for idx, filename in enumerate(image_files[:num_images]):
        print(f"\n[{idx+1}/{num_images}]")
        path = os.path.join(directory, filename)
        result = supervisor.plan_and_moderate(path)
        outputs.append(result)
        if result['overall'] == "SAFE":
            num_safe += 1
        logging.info(f"Processed {path}: Overall {result['overall']}")

    print("\n" + "=" * 60)
    print("📊 BATCH MODERATION COMPLETE")
    print(f"✅ Safe images: {num_safe}/{num_images} ({num_safe/num_images*100:.1f}%)")
    print(f"⚠️  Unsafe images: {num_images-num_safe}/{num_images} ({(num_images-num_safe)/num_images*100:.1f}%)")
    print("=" * 60)
    return outputs

if __name__ == "__main__":
    moderate_images(r"sample_data", num_images=10)