# Milestone 1: Dataset Exploration & Preprocessing

## 👥 Team Task Division (5 People)

---

## 🔹 Member 1 — Dataset Collection Lead

### Tasks:

**Select dataset**
- Prefer KITTI (lighter, easy for object detection)
- Or COCO / Open Images if you need more variety

**Download + organize the dataset into folders:**
```
/data/raw/images
/data/raw/labels
```

**Verify that bounding box annotations are available in formats such as:**
- KITTI .txt
- COCO .json
- VOC .xml

### Output:
- Structured dataset folder
- Document the dataset source + license

---

## 🔹 Member 2 — Data Exploration & Statistics

### Tasks:

**Count images + labels**

**Extract stats:**
- Number of objects per class
- Class imbalance chart
- Average bounding box sizes
- Image resolutions

**Visualizations:**
- Class frequency bar chart
- Example images with bounding boxes

**Check data issues:**
- Missing labels
- Empty images
- Very small bounding boxes
- Class imbalance (e.g., too many cars, few cyclists)

### Output:
- Plots + tables for the report
- Notes about issues

---

## 🔹 Member 3 — Data Quality & Environment Analysis

### Tasks:

**Inspect:**
- Day vs Night
- Weather (rain, fog, clear)
- Road type (highway, urban, rural)

**Look for biases:**
- E.g.: dataset has no nighttime pedestrians

**Prepare examples showing lighting/weather variation**

**Provide a small quality-review sample (e.g., 20 inspected images)**

### Output:
- Section of report describing environmental diversity
- Screenshots of example cases

---

## 🔹 Member 4 — Preprocessing & Augmentation Pipeline

### Tasks:

**Build a preprocessing pipeline using Python + OpenCV + Albumentations.**

**Steps:**
1. Resize images to 416×416 or 640×640
2. Normalize pixel values (0–1 or mean-std normalization)

**Augmentations:**
- Horizontal flip
- Random crop
- Rotation
- Brightness/contrast
- Blur (optional)
- Weather simulation (rain/fog) if needed

**Ensure bounding boxes transform correctly!**

### Output:
- A folder: `/data/preprocessed/`
- A Python notebook: `preprocessing_pipeline.ipynb`

---

## 🔹 Member 5 — Integration + Final Report Writer

### Tasks:

**Combine results from all 4 members**

**Create a professional PDF report (~5–7 pages):**
- Dataset description
- Class distributions (with charts)
- Image quality findings
- Environmental diversity
- Preprocessing summary
- Summary of any dataset limitations

**Upload preprocessed dataset + notebook to GitHub/Drive**

### Output:
- Dataset Exploration Report (PDF)
- Folder containing all preprocessed files
- Milestone 1 submission package

---

## 🧩 Recommended Tools

### For dataset handling:
- COCO API
- Roboflow (highly recommended — easy resizing and augmentation)
- Label Studio (if relabeling needed)

### For analysis:
- Python
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Albumentations (for augmentation)

---

## 📦 Final Deliverables (Exactly what the grader wants)

### 1️⃣ Dataset Exploration Report (PDF)

**Must include:**
- Introduction to dataset
- Total images + labels
- Class distribution chart
- Average bounding box sizes
- Image quality observations
- Lighting & weather diversity
- Dataset issues found
- Your preprocessing strategy

### 2️⃣ Preprocessed Dataset Folder
```
/data/preprocessed/images
/data/preprocessed/labels
```
- Matching bounding box annotations

---

## ⭐ My Recommendation

**Choose KITTI** for simplicity, fast preprocessing, and easy annotation structure.

**Use Roboflow** for:
- Automatic resizing
- Easy augmentation presets
- Exporting YOLO-ready data

*This will save your team at least 2–3 days.*

---

## 📘 Additional Resources Available

I can generate:
- ✅ A full PDF-style written report
- ✅ Python preprocessing code
- ✅ Notebook templates
- ✅ Class distribution charts

Just request any of these when ready!
