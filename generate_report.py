"""
generate_report.py
------------------
Generates classification report + confusion matrix using the ONNX model.
Run with:
    python generate_report.py
"""
import os, sys, glob, numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Windows: force UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/defect_efficientnet_b0.onnx"
# Tries each path; uses first one that has class sub-folders
DATA_DIRS   = [
    "src/serving/static/samples",
    "data/raw/train",
    "data/raw/val",
    "data/raw",
]
CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted_surface", "Rolled-in_scale", "Scratches"]
IMAGE_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_PER_CLASS = 100  # cap per class for speed; set to None for full eval

# ── Load ONNX model ───────────────────────────────────────────────────────────
print("Loading ONNX model...")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print(f"  Model loaded | input='{input_name}'")

# ── Find dataset root ─────────────────────────────────────────────────────────
data_root = None
for d in DATA_DIRS:
    if os.path.isdir(d):
        subdirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
        if len(subdirs) >= 2:
            data_root = d
            print(f"  Dataset root: {data_root}  ({len(subdirs)} class folders found)")
            break

if data_root is None:
    print("\nERROR: No dataset found in any of:", DATA_DIRS)
    print("Make sure you have images in class sub-folders.")
    sys.exit(1)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

# ── Run inference ─────────────────────────────────────────────────────────────
all_preds, all_labels = [], []
print("\nRunning inference...")

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    # Try exact match first, then case-insensitive
    cls_dir = os.path.join(data_root, cls_name)
    if not os.path.isdir(cls_dir):
        for d in os.listdir(data_root):
            if d.lower() == cls_name.lower():
                cls_dir = os.path.join(data_root, d)
                break

    if not os.path.isdir(cls_dir):
        print(f"  WARNING: '{cls_name}' folder not found, skipping")
        continue

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.bmp", "*.png", "*.tiff", "*.JPG", "*.BMP"):
        images.extend(glob.glob(os.path.join(cls_dir, ext)))
    images = list(set(images))
    if MAX_PER_CLASS:
        images = images[:MAX_PER_CLASS]

    if not images:
        print(f"  WARNING: No images in {cls_dir}")
        continue

    correct = 0
    for img_path in images:
        try:
            inp = preprocess(img_path)
            out = sess.run(None, {input_name: inp})[0][0]
            pred = int(np.argmax(softmax(out)))
            all_preds.append(pred)
            all_labels.append(cls_idx)
            if pred == cls_idx:
                correct += 1
        except Exception as e:
            pass

    acc = 100 * correct / len(images) if images else 0
    print(f"  {cls_name:<25}  {len(images):3d} images  accuracy={acc:.1f}%")

if not all_labels:
    print("\nERROR: No images were processed.")
    sys.exit(1)

# ── Classification Report ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES,
    digits=4
)
print(report)

with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Cloud-Native Automated Defect Detection System\n")
    f.write("EfficientNet-B0 (ONNX Runtime) | NEU Metal Surface Defect Dataset\n")
    f.write("="*60 + "\n\n")
    f.write(report)
print("  Saved: classification_report.txt")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
short_names = ["Crazing", "Inclusion", "Patches", "Pitted\nSurf.", "Rolled\nScale", "Scratches"]

fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
ax.set_title(
    "EfficientNet-B0  -  Confusion Matrix\nCloud-Native Automated Defect Detection System",
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
print("  Saved: confusion_matrix.png")
plt.close()

# ── Per-class Recall bar chart ────────────────────────────────────────────────
recalls = recall_score(
    all_labels, all_preds,
    average=None,
    labels=list(range(len(CLASS_NAMES))),
    zero_division=0
)
macro_recall = float(recalls.mean())

fig2, ax2 = plt.subplots(figsize=(9, 5))
colors = ['#2ecc71' if r >= 0.90 else '#e74c3c' for r in recalls]
bars = ax2.bar(short_names, recalls, color=colors, edgecolor='white', linewidth=0.8)
ax2.axhline(0.90, color='#e74c3c', linestyle='--', linewidth=1.5, label='Gate Threshold (0.90)')
ax2.axhline(macro_recall, color='#3498db', linestyle='-', linewidth=1.5,
            label=f'Macro Recall = {macro_recall:.4f}')
for bar, r in zip(bars, recalls):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{r:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 1.12)
ax2.set_ylabel('Recall', fontsize=12)
ax2.set_title('Per-Class Recall  |  EfficientNet-B0 on NEU Metal Surface Defect Dataset',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_facecolor('#f8f9fa')
fig2.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig("per_class_recall.png", dpi=150, bbox_inches='tight')
print("  Saved: per_class_recall.png")
plt.close()

print(f"\nDone!  Macro Recall = {macro_recall:.4f}")
print(f"  3 output files saved in: {os.getcwd()}")
print("  -> classification_report.txt")
print("  -> confusion_matrix.png")
print("  -> per_class_recall.png")
