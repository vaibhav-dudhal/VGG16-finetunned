import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import sys
import os
import time
import io
import zipfile
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ZIP_PATH             = r"C:\Users\Asus\OneDrive\Desktop\BTECH FINAL YEAR_Project\archive.zip"
NUM_CLASSES          = 2          # Real (client) = 0, Fake (imposter) = 1
BATCH_SIZE           = 32         # Optimized for RTX 3060 6GB VRAM
LR                   = 1e-4
MAX_EPOCHS           = 50
TARGET_ACC           = 99.99       # Target accuracy threshold
SAVE_PATH            = r"C:\Users\Asus\OneDrive\Desktop\VGG16\best_vgg16.pth"
LAST_CHECKPOINT_PATH = r"C:\Users\Asus\OneDrive\Desktop\VGG16\last_checkpoint.pth"
LOG_FILE_PATH        = r"C:\Users\Asus\OneDrive\Desktop\VGG16\training_log.txt"
FEATURE_EXCEL_PATH   = r"C:\Users\Asus\OneDrive\Desktop\VGG16\vgg16_features.xlsx"
ACCURACY_EXCEL_PATH  = r"C:\Users\Asus\OneDrive\Desktop\VGG16\dl_accuracy.xlsx"
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# DATASET — fast lazy-handle reading from ZIP
# ─────────────────────────────────────────────
class ZipFaceSpoofDataset(Dataset):
    def __init__(self, zip_path, txt_entries, label, transform=None):
        self.zip_path  = zip_path
        self.entries   = txt_entries
        self.label     = label
        self.transform = transform
        self.zf        = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, 'r')
        data = self.zf.read(self.entries[idx])
        img  = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label


def read_txt_from_zip(zf, txt_zip_path):
    """Read a split .txt inside the zip and return valid internal image paths."""
    with zf.open(txt_zip_path) as f:
        lines = f.read().decode("utf-8").splitlines()
    paths     = []
    all_names = set(zf.namelist())
    for line in lines:
        line = line.strip().replace('\r', '')
        if not line:
            continue
        if "raw/" in line:
            internal = "raw/" + line.split("raw/", 1)[1]
        else:
            internal = line
        if internal in all_names:
            paths.append(internal)
    return paths


# ─────────────────────────────────────────────
# VGG16 CONTINUOUS FEATURE EXTRACTOR (1048 values in [0, 1])
# ─────────────────────────────────────────────
class VGG16ContinuousFeatureExtractor(nn.Module):
    """
    Extracts 4096-dimensional fc2 activations, pools to 1048 values, then
    min-max normalizes each image's feature vector to [0, 1].
    This keeps continuous feature values rather than thresholding to 0/1.
    """
    def __init__(self, model, feature_dim=1048):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.fc = nn.Sequential(*list(model.classifier.children())[:5])
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dense_4096 = self.fc(x)
        pooled = nn.functional.adaptive_avg_pool1d(
            dense_4096.unsqueeze(1), self.feature_dim
        ).squeeze(1)
        minimum = pooled.min(dim=1, keepdim=True).values
        maximum = pooled.max(dim=1, keepdim=True).values
        span = maximum - minimum
        normalized = (pooled - minimum) / span.clamp_min(1e-8)
        return normalized.clamp(0.0, 1.0)



def get_feature_extractor(weights_path=SAVE_PATH, feature_dim=1048):
    """Creates and loads the VGG16 continuous feature extractor from a checkpoint."""
    base_model = models.vgg16(weights=None)
    base_model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=DEVICE)
        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        base_model.load_state_dict(state_dict)
    
    extractor = VGG16ContinuousFeatureExtractor(base_model, feature_dim=feature_dim)
    extractor = extractor.to(DEVICE)
    extractor.eval()
    return extractor



class ZipFeatureExportDataset(Dataset):
    def __init__(self, zip_path, entries_with_labels, transform=None):
        self.zip_path = zip_path
        self.entries_with_labels = entries_with_labels
        self.transform = transform
        self.zf = None

    def __len__(self):
        return len(self.entries_with_labels)

    def __getitem__(self, idx):
        path, label = self.entries_with_labels[idx]
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, "r")
        image = Image.open(io.BytesIO(self.zf.read(path))).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


def export_vgg_features_and_accuracy(model, val_loader, all_entries, val_tf):
    """Export continuous VGG features per image and DL training/validation accuracy."""
    import pandas as pd

    os.makedirs(os.path.dirname(FEATURE_EXCEL_PATH), exist_ok=True)
    extractor = VGG16ContinuousFeatureExtractor(model, feature_dim=1048).to(DEVICE).eval()
    export_loader = DataLoader(
        ZipFeatureExportDataset(ZIP_PATH, all_entries, transform=val_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    rows = []
    with torch.no_grad():
        for images, labels, paths in export_loader:
            images = images.to(DEVICE)
            feats = extractor(images).cpu().numpy()
            for path, label, vector in zip(paths, labels.numpy(), feats):
                # Standard label convention for exported workbooks: genuine=1, spoof=0.
                exported_label = 1 if int(label) == 0 else 0
                row = {"Image_Path": path, "Label": exported_label}
                row.update({f"VGG16_{i+1}": float(value) for i, value in enumerate(vector)})
                rows.append(row)
    pd.DataFrame(rows).to_excel(FEATURE_EXCEL_PATH, index=False, sheet_name="VGG16_Features")
    print(f"[+] Saved {len(rows)} images x 1048 continuous features: {FEATURE_EXCEL_PATH}")

    # The model's final held-out split accuracy is recorded in the training log/checkpoint.
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    accuracy_rows = [{
        "Metric": "Best validation accuracy (%)",
        "Value": float(ckpt.get("best_acc", ckpt.get("val_acc", 0.0)))
    }, {
        "Metric": "Best checkpoint epoch",
        "Value": int(ckpt.get("epoch", 0))
    }]
    pd.DataFrame(accuracy_rows).to_excel(ACCURACY_EXCEL_PATH, index=False, sheet_name="DL_Accuracy")
    print(f"[+] Saved DL accuracy summary: {ACCURACY_EXCEL_PATH}")


def main():
    print("=" * 60, flush=True)
    print(f"[*] Initializing VGG16 Anti-Spoofing Training", flush=True)
    print(f"   Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"   Batch Size: {BATCH_SIZE} | LR: {LR} | Target Acc: {TARGET_ACC}%", flush=True)
    print("=" * 60, flush=True)

    # ─────────────────────────────────────────────
    # LOAD SPLIT PATHS FROM ZIP
    # ─────────────────────────────────────────────
    print("Reading dataset split files from ZIP ...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        client_train   = read_txt_from_zip(zf, "raw/client_train_raw.txt")
        client_test    = read_txt_from_zip(zf, "raw/client_test_raw.txt")
        imposter_train = read_txt_from_zip(zf, "raw/imposter_train_raw.txt")
        imposter_test  = read_txt_from_zip(zf, "raw/imposter_test_raw.txt")

    print(f"  Train -> Real: {len(client_train)}, Fake: {len(imposter_train)} (Total: {len(client_train)+len(imposter_train)})", flush=True)
    print(f"  Test  -> Real: {len(client_test)}, Fake: {len(imposter_test)} (Total: {len(client_test)+len(imposter_test)})", flush=True)

    # ─────────────────────────────────────────────
    # TRANSFORMS
    # ─────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # ─────────────────────────────────────────────
    # BUILD DATALOADERS
    # ─────────────────────────────────────────────
    train_ds = ConcatDataset([
        ZipFaceSpoofDataset(ZIP_PATH, client_train,   label=0, transform=train_tf),
        ZipFaceSpoofDataset(ZIP_PATH, imposter_train, label=1, transform=train_tf),
    ])
    val_ds = ConcatDataset([
        ZipFaceSpoofDataset(ZIP_PATH, client_test,   label=0, transform=val_tf),
        ZipFaceSpoofDataset(ZIP_PATH, imposter_test, label=1, transform=val_tf),
    ])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    # ─────────────────────────────────────────────
    # MODEL
    #   Freeze : features[0..23]  (VGG16 blocks 1-4)
    #   Unfreeze: features[24..]  (block 5 — last conv block)
    #   Unfreeze: classifier       (fully connected layers)
    # ─────────────────────────────────────────────
    print("Loading VGG16 Pretrained Weights...", flush=True)
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze blocks 1-4
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze block 5 (conv5_1, conv5_2, conv5_3)
    for layer in model.features[24:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace final FC layer: 4096 -> 2 classes
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

    # Unfreeze entire classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model Ready: Trainable params: {trainable:,} / {total:,}", flush=True)

    # ─────────────────────────────────────────────
    # LOSS, OPTIMIZER, SCHEDULER
    # ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    # ─────────────────────────────────────────────
    # RESUME FROM CHECKPOINT OR START FRESH
    # ─────────────────────────────────────────────
    start_epoch = 1
    best_acc    = 0.0

    ckpt_path = None
    if os.path.exists(LAST_CHECKPOINT_PATH):
        ckpt_path = LAST_CHECKPOINT_PATH
    elif os.path.exists(SAVE_PATH):
        ckpt_path = SAVE_PATH

    if ckpt_path:
        print(f"[*] Found checkpoint at: {ckpt_path}", flush=True)
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc    = checkpoint.get('best_acc', checkpoint.get('val_acc', 0.0))
            print(f"[*] Successfully resumed from Epoch {start_epoch} (Best Val Acc so far: {best_acc:.2f}%)", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not load checkpoint ({e}). Starting fresh.", flush=True)
    else:
        print("[*] No existing checkpoint found. Starting fresh training.", flush=True)

    # ─────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # ── Train phase ──
        model.train()
        running_loss, correct, total_samples = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item() * imgs.size(0)
            _, preds       = outputs.max(1)
            correct       += preds.eq(labels).sum().item()
            total_samples += imgs.size(0)

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = 100.0 * correct / total_samples
                print(f"  [Epoch {epoch:02d}] Batch {batch_idx+1:03d}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | Running Acc: {batch_acc:.2f}%", flush=True)

        train_loss = running_loss / total_samples
        train_acc  = 100.0 * correct / total_samples

        # ── Validation phase ──
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs      = model(imgs)
                loss         = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                _, preds     = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += imgs.size(0)

        val_acc  = 100.0 * val_correct / val_total
        val_loss = val_loss / val_total
        scheduler.step(val_acc)
        elapsed = time.time() - epoch_start

        print(f"\n Epoch [{epoch:02d}/{MAX_EPOCHS}] Finished in {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%\n", flush=True)

        # ── Check if best model ──
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save({
                'epoch'          : epoch,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_acc'        : val_acc,
                'val_loss'       : val_loss,
                'best_acc'       : best_acc,
            }, SAVE_PATH)
            print(f"   [BEST] Best model saved! Val Acc: {val_acc:.2f}%\n", flush=True)

        # ── Always save last checkpoint for safety & auto-resume ──
        torch.save({
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_loss'     : train_loss,
            'train_acc'      : train_acc,
            'val_loss'       : val_loss,
            'val_acc'        : val_acc,
            'best_acc'       : best_acc,
        }, LAST_CHECKPOINT_PATH)

        # ── Append to persistent log file ──
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                    f"Best: {best_acc:.2f}% | Time: {elapsed:.1f}s | "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ── Early stop on target ──
        if val_acc >= TARGET_ACC:
            print(f"[*] Target accuracy of {TARGET_ACC}% reached at epoch {epoch}!", flush=True)
            break

    # Export features for every image listed in the train/test splits.
    all_entries = (
        [(p, 0) for p in client_train] + [(p, 1) for p in imposter_train] +
        [(p, 0) for p in client_test] + [(p, 1) for p in imposter_test]
    )
    # Remove duplicate paths while preserving order.
    seen = set()
    all_entries = [(p, y) for p, y in all_entries if not (p in seen or seen.add(p))]
    export_vgg_features_and_accuracy(model, val_loader, all_entries, val_tf)

    print(f"\n[DONE] Training complete! Best Val Accuracy: {best_acc:.2f}%", flush=True)
    print(f"[PATH] Checkpoint: {SAVE_PATH}", flush=True)


if __name__ == '__main__':
    main()