# Segformer 을 이용한 차선인식 프로젝트
<프로젝트 참고서>
https://docs.google.com/document/d/1rxQHvxAIZM0pTspVIDAUx2ZEmjZRNSW-f6OrcitXFqM/edit?tab=t.0

## 1. 로보플로우에서 라밸링 
<img width="1902" height="883" alt="화면 캡처 2025-08-13 111412" src="https://github.com/user-attachments/assets/d4db8502-e261-4cdc-994c-ba411257a9a9" />

## 2. 라벨링한 데이터셋을 Image Mask 형식으로 저장
<img width="1566" height="877" alt="화면 캡처 2025-08-13 111834" src="https://github.com/user-attachments/assets/55a8e2df-364b-4f54-99e0-ab0f5f571c41" />

## 3. 저장한 데이터셋 파일을 가지고 Colab에 가서 전이학습
```
# ========================
# 0) 필수 라이브러리 설치
# ========================
!pip install -q "transformers>=4.44,<5" accelerate evaluate opencv-python-headless pillow

# ========================
# 1) Roboflow ZIP 업로드
# ========================
from google.colab import files
up = files.upload()  # Roboflow에서 받은 dataset.zip 선택
ZIP_PATH = "/content/" + list(up.keys())[0]

# ========================
# 2) 압축 해제
# ========================
import os, zipfile, shutil

EXTRACT_DIR = "/content/ds_rf"
if os.path.isdir(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

print("unzipped to", EXTRACT_DIR, "->", os.listdir(EXTRACT_DIR))

# ========================
# 3) 데이터 구조 파악
# ========================
def find_split_dir(root, names=("train","valid","val","test")):
    found = {}
    for n in names:
        p = os.path.join(root, n)
        if os.path.isdir(p):
            found["valid" if n in ("valid","val") else n] = p
    return found

splits = find_split_dir(EXTRACT_DIR)
if not splits:
    raise RuntimeError("train/valid/test 폴더를 찾지 못함. ZIP 내용 확인")

# ========================
# 4) 학습 클래스 설정
# ========================
COLLAPSE_TO_BINARY = True  # True면 모든 non-zero → 'lane(1)'
if COLLAPSE_TO_BINARY:
    CLASS_NAMES = ["background", "lane"]
else:
    CLASS_NAMES = ["background", "lane", "lane-dot", "lane-mid", "lane_crosswalk"]

id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
label2id = {n: i for i, n in id2label.items()}
NUM_LABELS = len(CLASS_NAMES)

# === 패치: RFSegFolder를 더 관대한 버전으로 재정의 ===
import os, glob, re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 이미지/마스크 파일명 매칭을 위해 뒤에 붙는 접미어들을 제거
_SUFFIX_RE = re.compile(r'(_|-)(mask|masks|label|labels|seg|segment|segmentation)$', re.I)

def _stem_no_suffix(path):
    s = os.path.splitext(os.path.basename(path))[0]
    s = _SUFFIX_RE.sub('', s)   # ..._mask, -labels 등 제거
    return s

def _is_img(name):
    return name.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))

class RFSegFolder(Dataset):
    def __init__(self, split_dir, processor):
        # 1) 이미지 폴더 탐색: 'images/'가 있으면 거기, 없으면 split 루트에서 바로 찾기
        img_cands = [os.path.join(split_dir, "images"), split_dir]
        self.img_dir = None
        for d in img_cands:
            if os.path.isdir(d) and any(_is_img(f) for f in os.listdir(d)):
                self.img_dir = d
                break
        if self.img_dir is None:
            raise RuntimeError(f"No images found in {split_dir}")

        # 2) 마스크 폴더 후보: labels/masks/annotations/… 없으면 split 루트까지 포함
        mask_cands = ["masks","labels","annotations","masks_png","labels_png","mask","Labels","Masks"]
        self.mask_dirs = [os.path.join(split_dir, c) for c in mask_cands if os.path.isdir(os.path.join(split_dir, c))]
        if not self.mask_dirs:
            # 마지막 수단: split 디렉토리 안에서 PNG가 있는 모든 폴더를 스캔(이미지 폴더 제외)
            self.mask_dirs = []
            for root, dirs, files in os.walk(split_dir):
                if os.path.abspath(root) == os.path.abspath(self.img_dir):
                    continue
                if any(f.lower().endswith(".png") for f in files):
                    self.mask_dirs.append(root)
            if not self.mask_dirs:
                # 정말 없으면 루트도 후보에 포함(아주 드문 케이스)
                self.mask_dirs = [split_dir]

        self.processor = processor

        # 3) 마스크 인덱스 구축 (동일 stem 매칭)
        mask_map = {}
        for md in self.mask_dirs:
            for p in glob.glob(os.path.join(md, "*.png")):
                mask_map[_stem_no_suffix(p)] = p

        # 4) 이미지-마스크 페어 만들기
        self.items = []
        for ip in sorted(glob.glob(os.path.join(self.img_dir, "*.*"))):
            if not _is_img(ip):
                continue
            st = _stem_no_suffix(ip)
            mp = mask_map.get(st)
            if mp and os.path.exists(mp):
                self.items.append((ip, mp))

        if not self.items:
            # 디버깅 도움: 폴더 안에 뭐가 있는지 조금 찍어줌
            print("[DEBUG] img_dir:", self.img_dir)
            print("[DEBUG] mask_dirs:", self.mask_dirs[:3], "…", f"({sum(len(glob.glob(os.path.join(d,'*.png'))) for d in self.mask_dirs)} masks png)")
            raise RuntimeError(f"No (image,mask) pairs in {split_dir}. "
                               f"이미지/마스크 파일명이 서로 매칭되는지(예: abc.jpg ↔ abc_mask.png) 확인해주세요.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        image = Image.open(ip).convert("RGB")
        # 팔레트/그레이스케일 모두 지원: 0=배경, 1+=전부 차선으로 뭉치기(이진)
        m = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
        m = (m > 0).astype(np.uint8)  # 이진 세팅 (여러 클래스를 쓰려면 여기 로직 바꿔도 됨)
        enc = processor(images=image, segmentation_maps=m, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

# ========================
# 5) 프로세서/모델 로드
# ========================
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

CKPT = "nvidia/segformer-b0-finetuned-ade-512-512"

processor = SegformerImageProcessor.from_pretrained(
    CKPT,
    reduce_labels=False  # 라벨 줄임 비활성화(우리 클래스 인덱스 유지)
)

model = SegformerForSemanticSegmentation.from_pretrained(
    CKPT,
    num_labels=NUM_LABELS,   # 위에서 만든 NUM_LABELS 사용
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # 클라스 수가 달라 생기는 shape mismatch 허용
)

# ========================
# 6) 데이터셋 생성
# ========================
# splits는 이미 위에서 만들어 둔 dict: {'train': ..., 'valid': ..., 'test': ...} 중 일부
train_dir = splits.get("train")
valid_dir = splits.get("valid") or splits.get("val") or train_dir  # valid 없으면 train 재사용

if train_dir is None:
    raise RuntimeError("train 폴더를 찾지 못했습니다. ZIP 구조를 확인하세요.")

train_ds = RFSegFolder(train_dir, processor)  # 이진 세팅은 클래스 내부에서 m>0 → 1로 처리
val_ds   = RFSegFolder(valid_dir, processor)
print(f"✅ Dataset ready: train={len(train_ds)}, valid={len(val_ds)}")

# ========================
# 7) 학습 + 저장 + 다운로드
# ========================
from transformers import TrainingArguments, Trainer
import numpy as np, evaluate, torch, os, shutil, zipfile
from google.colab import files

metric = evaluate.load("mean_iou")

def _to_py(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (N, C, h, w), labels: (N, H, W)
    if isinstance(logits, tuple):
        logits = logits[0]
    lt = torch.from_numpy(logits)
    yt = torch.from_numpy(labels)

    # 라벨 크기에 맞춰 업샘플(크기 불일치 방지)
    lt_up = torch.nn.functional.interpolate(
        lt, size=yt.shape[-2:], mode="bilinear", align_corners=False
    )
    preds = lt_up.argmax(dim=1).cpu().numpy()

    res = metric.compute(
        predictions=preds,
        references=labels,
        num_labels=NUM_LABELS,
        ignore_index=255,
        reduce_labels=False,
    )
    return {k: _to_py(v) for k, v in res.items()}

args = TrainingArguments(
    output_dir="segformer-lane",
    learning_rate=5e-5,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch", # Added comma here
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# 학습
trainer.train()

# 베스트 저장
BEST_DIR = "segformer-lane/best"
os.makedirs(BEST_DIR, exist_ok=True)
trainer.save_model(BEST_DIR)            # 모델 가중치/구성
processor.save_pretrained(BEST_DIR)     # 프로세서

print("✅ Saved best to:", BEST_DIR)

# 아티팩트 압축(zip) 후 다운로드(필요한 것만 묶음)
ZIP_OUT = "segformer_lane_best.zip"
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
    # 핵심 파일들만 선택 저장
    for fname in [
        "config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"
    ]:
        p = os.path.join(BEST_DIR, fname)
        if os.path.exists(p):
            z.write(p, arcname=os.path.join("best", fname))
    # trainer args/로그 등 메타(선택)
    for extra in ["trainer_state.json", "trainer_config.json", "all_results.json"]:
        p = os.path.join("segformer-lane", extra)
        if os.path.exists(p):
            z.write(p, arcname=os.path.join("run_meta", extra))

print("📦 Zip created:", ZIP_OUT)

# 다운로드 트리거
files.download(ZIP_OUT)  # Colab에서 파일 다운
```

## 4. 영상 업로드 코드 쉘 추가
```
# ▶ 영상 업로드
from google.colab import files
up = files.upload()  # 로컬에서 mp4 등 선택
VIDEO_IN = "/content/" + list(up.keys())[0]
print("입력 영상:", VIDEO_IN)

# ▶ SegFormer 추론 + 컬러 오버레이
import os, cv2, json, numpy as np, torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ===== 경로 설정 =====
MODEL_DIR = "/content/segformer-lane/best"  # 학습 저장한 디렉터리
VIDEO_OUT = "/content/out_lane_overlay.mp4"
ALPHA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 모델/프로세서 로드 =====
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()

# ===== 라벨 이름 & 색상 고정 =====
label_names = ["background", "lane", "lane_dot", "lane_mid", "lane_crosswalk"]

# OpenCV BGR 색상 (배경 제외)
PALETTE = [
    (0, 0, 0),           # background
    (0, 255, 0),         # lane         - 초록
    (0, 165, 255),       # lane_dot     - 주황
    (255, 0, 0),         # lane_mid     - 파랑
    (255, 255, 255)      # crosswalk    - 흰색
]
num_labels = len(PALETTE)

print("Label names:", label_names)
print("Num labels:", num_labels)

# ===== 비디오 IO =====
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# ===== 추론 루프 =====
with torch.no_grad():
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

        outputs = model(**inputs)
        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        # ===== 컬러 오버레이 =====
        overlay = np.zeros_like(frame)
        for cls_id in range(1, num_labels):  # 0=배경은 제외
            mask = (pred == cls_id)
            if mask.any():
                overlay[mask] = PALETTE[cls_id]

        blended = cv2.addWeighted(frame, 1.0 - ALPHA, overlay, ALPHA, 0.0)
        out.write(blended)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

cap.release()
out.release()
print("✅ Saved video:", VIDEO_OUT)

# 결과 다운로드
files.download(VIDEO_OUT)
```

## 5. 결과 영상 확인
--> 차선 인식은 잘 되지만 실선과 점선 색깔 구분이 안되는 아쉬움이 있다.
<img width="1917" height="930" alt="화면 캡처 2025-08-13 112305" src="https://github.com/user-attachments/assets/a1f63f0b-d619-493c-9431-45a7bf8ec141" />

