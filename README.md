# Vehicle Counting with CLIP

An experimental computer vision project that investigates whether **CLIP (Contrastive Language–Image Pretraining)** can estimate the number of vehicles visible in an image.

The repository compares three approaches:

1. Zero-shot vehicle counting with OpenAI CLIP
2. Prompt-engineered counting with Hugging Face CLIP
3. CLIP image embeddings combined with a trained Logistic Regression classifier

> **Note:** CLIP is designed for image–text similarity, not precise object counting. This project is therefore an experiment that compares different ways of adapting CLIP to a counting task.

## Project Goals

The main goals of this project are to:

- Explore CLIP for vehicle-count prediction
- Compare zero-shot and trained classification approaches
- Test prompt engineering and image augmentation
- Evaluate prediction confidence and counting accuracy
- Demonstrate the limitations of vision-language models for object counting

## Approaches

### 1. Baseline CLIP Zero-Shot Counting

File: `CLIPCounter.py`

This approach uses OpenAI CLIP directly without training an additional model.

For every image, the model compares its image embedding with text prompts representing different vehicle counts, such as:

```text
A traffic camera photo showing exactly 0 passenger cars
A traffic camera photo showing exactly 1 passenger car
A traffic camera photo showing exactly 2 passenger cars
```

Several prompt variations are generated for each possible count. Their text embeddings are averaged and compared with the image embedding.

The predicted vehicle count is the prompt class with the highest similarity score.

### 2. Hugging Face CLIP with Prompt Engineering

File: `CLIPCounterHuggingFace.py`

This approach uses a CLIP model from Hugging Face and expands the prompt-based method with:

- Multiple prompt templates
- Confidence measurements
- Probability thresholds
- Image augmentation
- Additional prediction calibration

Although the model may produce higher confidence values, higher confidence does not necessarily mean better counting accuracy.

### 3. CLIP Embeddings with Logistic Regression

File: `CLIPCounterHFTrain.py`

This approach uses CLIP as a feature extractor.

The process is:

1. Generate an embedding for every image with CLIP.
2. Use the known vehicle counts as labels.
3. Train a Logistic Regression classifier on the embeddings.
4. Predict the vehicle-count class for unseen images.
5. Evaluate the predictions against the ground-truth counts.

This approach produces the best overall results among the three implementations because the classifier learns directly from labelled examples instead of relying only on image–text similarity.

## Image Augmentation

The project uses simple augmentation by processing both:

- The original image
- A horizontally mirrored version of the image

The embeddings from the augmented images are combined to create a more stable representation.

## Dataset

The scripts work with image metadata and object annotations stored in JSON files.

The annotations are filtered using vehicle-related labels such as:

- Car
- Automobile
- Sedan
- SUV
- Van
- Taxi

The filtered annotations provide the ground-truth vehicle counts used during evaluation and classifier training.

The repository also contains:

| File | Description |
|---|---|
| `train.csv` | Training data containing image paths and vehicle-count labels |
| `vgc_clip_vehicle_counts.csv` | Results produced by the baseline CLIP approach |
| `vgh_clip_vehicle_counts.csv` | Results produced by the Hugging Face prompt-based approach |
| `vg_clip_vehicle_counts.csv` | Results produced by the trained CLIP classifier |
| `YOLODraw.py` | Utility for drawing or visualizing detected objects |
| `experimental/` | Additional experimental scripts and files |

## Project Structure

```text
vehicle-counter/
├── experimental/
├── CLIPCounter.py
├── CLIPCounterHuggingFace.py
├── CLIPCounterHFTrain.py
├── YOLODraw.py
├── train.csv
├── vg_clip_vehicle_counts.csv
├── vgc_clip_vehicle_counts.csv
├── vgh_clip_vehicle_counts.csv
└── README.md
```

## Technologies

- Python
- PyTorch
- OpenAI CLIP
- Hugging Face Transformers
- scikit-learn
- pandas
- Pillow
- NumPy
- Requests
- Joblib

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/veronika-ilioska/vehicle-counter.git
cd vehicle-counter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install torch torchvision transformers scikit-learn pandas pillow numpy requests joblib
pip install git+https://github.com/openai/CLIP.git
```

## Configuration

Before running the scripts, update the local paths used for:

- Object annotation JSON file
- Image metadata JSON file
- Downloaded image directory
- Training CSV file
- Output CSV files

For example:

```python
OBJECTS_JSON = Path("data/objects.json")
IMAGES_JSON = Path("data/image_data.json")
IMG_OUT_DIR = Path("data/images")
```

The current scripts may contain absolute Windows paths, so they must be changed to match your local project structure.

A recommended structure is:

```text
vehicle-counter/
├── data/
│   ├── objects.json
│   ├── image_data.json
│   └── images/
├── CLIPCounter.py
├── CLIPCounterHuggingFace.py
└── CLIPCounterHFTrain.py
```

## Running the Project

### Run the baseline OpenAI CLIP approach

```bash
python CLIPCounter.py
```

### Run the Hugging Face prompt-based approach

```bash
python CLIPCounterHuggingFace.py
```

### Train and evaluate the CLIP classifier

```bash
python CLIPCounterHFTrain.py
```

The scripts generate CSV files containing information such as:

- Image ID
- Ground-truth vehicle count
- Predicted count
- Confidence score
- Maximum softmax probability
- Image path

## Evaluation

The three approaches can be compared using metrics such as:

- Exact count accuracy
- Mean absolute error
- Prediction confidence
- Confusion matrix
- Per-class accuracy

The trained CLIP embedding classifier performs better than the zero-shot prompt-based approaches, but its performance is still limited by:

- Small or imbalanced training data
- Crowded traffic scenes
- Occluded vehicles
- Very small vehicles
- Similar-looking non-vehicle objects
- CLIP's limited ability to represent exact quantities

## Limitations

CLIP learns broad visual-semantic relationships and is not designed to locate and count individual objects.

For a production vehicle-counting system, an object detector and tracker would usually be more suitable, such as:

- YOLO
- Faster R-CNN
- RetinaNet
- DETR
- Deep SORT or ByteTrack for video tracking

This repository focuses on experimentation and comparison rather than production-level vehicle counting.


## Author

**Veronika Ilioska**

GitHub: [veronika-ilioska](https://github.com/veronika-ilioska)

## License

This project was created for educational and experimental purposes.
