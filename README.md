# VirtuShield: Multi-Agent VR Content Moderation System

⚠️ **IMPORTANT WARNING** ⚠️
**This repository contains explicit adult content in the `sample_data` folder for research and testing purposes. These images include nudity and other potentially offensive material. Please be aware of this before cloning, browsing, or working with this repository.**

## Overview

VirtuShield is a multi-agent system for moderating virtual reality content using computer vision and natural language processing. The system employs a hierarchical architecture with specialized agents for detection, assessment, and supervision to identify potentially inappropriate content in VR environments.

## Features

- **Multi-Agent Architecture**: Hierarchical system with Detection, NSFW Assessment, and Supervisor agents
- **Hybrid Approach**: Combines YOLO object detection with CLIP-based content classification
- **Meta-Learning Enhancement**: Utilizes GPT-4 for iterative prompt refinement in borderline cases
- **Ablation Study Support**: Built-in toggles for reflection and nudity detection components
- **Batch Processing**: Efficient processing of multiple images with detailed reporting

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM
- Operating System: Windows, Linux, or macOS

## Installation

### 1. Clone the Repository

```bash
git clone [repository-url]
cd virtushield
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, install the following packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install transformers
pip install pillow
pip install numpy
pip install openai
pip install python-dotenv
```

### 4. Model Setup

#### YOLO Model
The YOLO model weights (`yolo_ft.pt`) are included in the repository. The model is automatically loaded from the root directory.

#### CLIP Model
The CLIP model will be automatically downloaded from Hugging Face on first run.

### 5. Environment Configuration

Create a `.env` file in the project root directory:

```bash
OPENAI_API=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Usage

### Basic Usage

Run the moderation pipeline on sample images:

```bash
python moderation_pipeline_v4.py
```

By default, this will process 10 images from the `sample_data` directory.

### Custom Configuration

Modify the main execution block in `moderation_pipeline_v4.py`:

```python
if __name__ == "__main__":
    # Process all images in a directory
    moderate_images("path/to/your/images", num_images=-1)
    
    # Process specific number of images
    moderate_images("path/to/your/images", num_images=20)
```

### Ablation Studies

The system includes toggles for ablation studies:

```python
# Toggle reflection mechanism (line 13)
reflection_active = True  # Set to False to disable reflection

# Toggle YOLO nudity detection (line 14)
activate_yolo = True  # Set to False to disable YOLO NSFW labels
```

## Input Data Format

### Image Requirements
- Supported formats: PNG, JPG, JPEG
- Place images in a directory (default: `sample_data/`)
- No specific resolution requirements (images will be processed as-is)

### Directory Structure
```
virtushield/
├── moderation_pipeline_v4.py
├── yolo_ft.pt
├── requirements.txt
├── .env
├── sample_data/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── README.md
```

## Output

The system generates detailed console output including:
- Detection results for each region
- NSFW assessment probabilities
- Iterative refinement steps (when triggered)
- Final classification (SAFE/UNSAFE)
- Batch processing statistics

Results are returned as structured dictionaries containing:
- Image path
- Detected regions with bounding boxes
- NSFW classification and confidence scores
- Overall safety verdict

## Configuration Parameters

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reflection_active` | Enable/disable meta-LLM reflection | `True` |
| `activate_yolo` | Enable/disable YOLO NSFW detection | `True` |
| `max_iterations` | Maximum reflection iterations | `3` |
| `borderline_threshold` | Probability range for borderline cases | `0.4 - 0.7` |

### Model Selection

The system uses:
- **YOLO**: Custom-trained model for object and NSFW content detection
- **CLIP**: `openai/clip-vit-base-patch32` for visual-language understanding
- **GPT-4**: `gpt-4o-mini` for meta-prompt refinement

## Troubleshooting

### Common Issues

1. **CUDA not available**: System will automatically fall back to CPU processing
2. **OpenAI API errors**: Verify API key in `.env` file and check quota/limits
3. **Model loading errors**: Ensure YOLO weights path is correct
4. **Memory issues**: Reduce batch size or process fewer images at once

### Performance Tips

- Use GPU acceleration for faster processing
- Adjust `max_iterations` to balance accuracy vs. speed
- Process images in batches for efficiency
- Monitor API usage to avoid rate limits

