"""Dataset loading for ImageNet-C, ImageNet-A, ImageNet-R, and ImageNet-style folders."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}

DATASET_PATHS = {
    "imagenet-c": ["imagenet-c", "ImageNet-C", "imagenet_C"],
    "imagenet-a": ["imagenet-a", "ImageNet-A", "imagenet_a"],
    "imagenet-r": ["imagenet-r", "ImageNet-R", "imagenet_r"],
    "images_largescale": ["images_largescale", "ImageNet", "imagenet", "ILSVRC2012", "ILSVRC2012_img_val"],
    "cifar100-c": ["cifar100-c/CIFAR-100-C", "CIFAR-100-C", "cifar100-c"],
}

IMAGENET_LABELED_DATASETS = {"imagenet-c", "imagenet-a", "imagenet-r", "images_largescale", "cifar100-c"}
SUPPORTED_DATASETS = ("imagenet-c", "imagenet-a", "imagenet-r", "cifar100-c")


def resolve_dataset_path(data_root: str, dataset: str) -> Path:
    dataset = dataset.lower()
    root = Path(data_root)
    rels = DATASET_PATHS.get(dataset, [dataset])
    candidates = [root / rel for rel in rels]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _folder_names(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _numeric_class_mapping(root: Path) -> Dict[str, int]:
    names = _folder_names(root)
    if names and all(n.isdigit() for n in names):
        return {n: int(n) for n in names}
    return {}


def _requires_imagenet_mapping(dataset: str, root: Path) -> bool:
    if dataset.lower() not in IMAGENET_LABELED_DATASETS:
        return False
    names = _folder_names(root)
    if not names:
        return False
    return any(n.startswith("n") and len(n) >= 8 for n in names)


def _sorted_wnid_fallback_mapping(root: Path) -> Dict[str, int]:
    """Last-resort mapping for full 1000-class folder trees.

    This fallback is intentionally restricted to near-complete ImageNet trees. It is
    not used for ImageNet-A/R subsets because evaluating those datasets with a
    sequential or subset-sorted mapping would silently produce wrong accuracies.
    """
    names = _folder_names(root)
    wnids = [n for n in names if n.startswith("n") and len(n) >= 8]
    if len(wnids) >= 900 and len(wnids) == len(names):
        return {wnid: idx for idx, wnid in enumerate(sorted(wnids))}
    return {}


def _mapping_for_root(dataset: str, root: Path, imagenet_mapping: Optional[Dict[str, int]]):
    numeric = _numeric_class_mapping(root)
    if numeric:
        return numeric

    imagenet_mapping = imagenet_mapping or {}
    names = _folder_names(root)
    if imagenet_mapping:
        ok = sum(1 for n in names if n in imagenet_mapping)
        if ok == len(names) or ok >= 900:
            return imagenet_mapping

    fallback = _sorted_wnid_fallback_mapping(root)
    if fallback:
        return fallback

    if _requires_imagenet_mapping(dataset, root):
        missing = sorted([n for n in names if n.startswith("n") and n not in imagenet_mapping])[:10]
        raise RuntimeError(
            f"{dataset} uses ImageNet WordNet-ID folders under {root}, but a correct wnid->ImageNet-1K "
            f"index mapping was not found. Missing examples: {missing}. Put imagenet_class_index.json, "
            f"ILSVRC2012_class_index.json, synset_words.txt, or torchvision ImageNet meta.bin under "
            f"{Path(root).anchor or '/'}data/share/datasets (normally /data/share/datasets), or install a timm "
            "version exposing ImageNetInfo. This check is deliberate: using subset folder order for ImageNet-A/R "
            "would make accuracy values wrong."
        )
    return {}


def imagenet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def list_image_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in IMG_EXTS]


class ImageListDataset(Dataset):
    def __init__(self, root: str, transform=None, label: int = -1, is_ood: int = 0):
        self.root = Path(root)
        self.files = sorted(list_image_files(self.root))
        self.transform = transform
        self.label = int(label)
        self.is_ood = int(is_ood)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label, self.is_ood, str(path)


class WrappedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, class_to_imagenet_idx: Optional[Dict[str, int]] = None, is_ood=0):
        super().__init__(root=root, transform=transform)
        self.samples = sorted(self.samples, key=lambda x: x[0])
        self.imgs = self.samples
        self.is_ood = int(is_ood)
        self.mapping_status = "sequential_folder_indices"
        if class_to_imagenet_idx:
            mapped_samples = []
            missing = []
            for path, _target in self.samples:
                cls = Path(path).parent.name
                if cls not in class_to_imagenet_idx:
                    missing.append(cls)
                    continue
                mapped_samples.append((path, int(class_to_imagenet_idx[cls])))
            if missing:
                raise RuntimeError(f"Missing class mapping for {len(set(missing))} class folders, e.g. {sorted(set(missing))[:10]}")
            self.samples = mapped_samples
            self.imgs = self.samples
            self.targets = [t for _, t in self.samples]
            self.mapping_status = "imagenet_synset_or_numeric_mapping"

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return img, int(target), self.is_ood, self.samples[index][0]


def _load_mapping_json(path: Path) -> Dict[str, int]:
    obj = json.loads(path.read_text())
    mapping = {}
    # Keras/torchvision style: {"0": ["n01440764", "tench"], ...}
    for k, v in obj.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if isinstance(v, list) and v:
            mapping[str(v[0])] = idx
        elif isinstance(v, str):
            mapping[v.split()[0]] = idx
    return mapping


def _load_meta_mat(path: Path) -> Dict[str, int]:
    """Load ILSVRC devkit meta.mat if scipy is available."""
    try:
        import scipy.io as sio
    except Exception:
        return {}
    try:
        meta = sio.loadmat(str(path), squeeze_me=True)["synsets"]
        rows = []
        for item in meta:
            ilsvrc_id = int(item[0])
            wnid = str(item[1])
            rows.append((ilsvrc_id, wnid))
        rows = sorted(rows, key=lambda x: x[0])
        mapping = {wnid: idx for idx, (_, wnid) in enumerate(rows[:1000])}
        return mapping if len(mapping) >= 900 else {}
    except Exception:
        return {}


def load_imagenet_class_mapping(data_root: str) -> Dict[str, int]:
    """Load WordNet-ID -> ImageNet-1K index mapping from local sources.

    Correct mapping is essential for ImageNet-A/R because their folder trees contain
    a subset of ImageNet WordNet IDs; sequential ImageFolder labels would be wrong.
    """
    root = Path(data_root)
    candidates = [
        root / "imagenet_class_index.json",
        root / "ILSVRC2012_class_index.json",
        root / "imagenet_class_index.txt",
        root / "synset_words.txt",
        root / "ImageNet" / "meta.bin",
        root / "imagenet" / "meta.bin",
        root / "images_largescale" / "meta.bin",
        root / "ILSVRC2012_devkit_t12" / "data" / "meta.mat",
        root / "ILSVRC2012_devkit_t3" / "data" / "meta.mat",
        root / "meta.mat",
    ]

    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix == ".json":
                mapping = _load_mapping_json(p)
                if len(mapping) >= 900:
                    return mapping
            elif p.name == "meta.bin":
                obj = torch.load(str(p), map_location="cpu")
                if isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[1], list):
                    mapping = {str(wnid): idx for idx, wnid in enumerate(obj[1])}
                    if len(mapping) >= 900:
                        return mapping
            elif p.name == "meta.mat":
                mapping = _load_meta_mat(p)
                if len(mapping) >= 900:
                    return mapping
            else:
                mapping = {}
                for idx, line in enumerate(p.read_text(errors="ignore").splitlines()):
                    parts = line.strip().split()
                    if parts and parts[0].startswith("n"):
                        mapping[parts[0]] = idx
                if len(mapping) >= 900:
                    return mapping
        except Exception:
            continue

    try:
        from timm.data.imagenet_info import ImageNetInfo

        info = ImageNetInfo()
        for attr in ["_synset_to_idx", "synset_to_idx", "wnid_to_idx"]:
            maybe = getattr(info, attr, None)
            if isinstance(maybe, dict) and len(maybe) >= 900:
                return {str(k): int(v) for k, v in maybe.items()}
        index_to_synset = getattr(info, "index_to_synset", None)
        if callable(index_to_synset):
            mapping = {str(index_to_synset(i)): i for i in range(1000)}
            if len(mapping) >= 900:
                return mapping
    except Exception:
        pass

    return {}


def _parse_severities(severity) -> List[int]:
    if severity in ["all", None, ""]:
        return [1, 2, 3, 4, 5]
    return [int(s) for s in str(severity).split(",") if str(s).strip()]


def discover_imagenet_c_combos(data_root: str, corruption="all", severity="all") -> List[Tuple[str, int, Path]]:
    root = resolve_dataset_path(data_root, "imagenet-c")
    if not root.exists():
        return []

    corruptions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if corruption and corruption != "all":
        wanted = [c.strip() for c in str(corruption).split(",") if c.strip()]
        corruptions = [c for c in corruptions if c in wanted]

    severities = _parse_severities(severity)
    combos = []
    for c in corruptions:
        for s in severities:
            p = root / c / str(s)
            if p.exists():
                combos.append((c, s, p))
    return combos


# ---------------------------------------------------------------------------
# CIFAR-100-C support (numpy-array based corruption benchmark)
# ---------------------------------------------------------------------------
CIFAR100_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "saturate", "spatter",
]


class CIFAR100CDataset(Dataset):
    """CIFAR-100-C corruption dataset (numpy arrays, 32x32x3 uint8).

    The CIFAR-100-C archive stores each corruption as a single .npy file of
    shape (50000, 32, 32, 3) uint8 covering all 5 severities concatenated
    (10000 per severity). The labels.npy file provides the ground-truth
    class indices (0-99).
    """

    SAMPLES_PER_SEVERITY = 10000

    def __init__(self, data_root: str, corruption: str, severity: int,
                 transform=None):
        root = resolve_dataset_path(data_root, "cifar100-c")
        if not root.exists():
            raise FileNotFoundError(f"CIFAR-100-C root not found: {root}")
        corr_path = root / f"{corruption}.npy"
        label_path = root / "labels.npy"
        if not corr_path.exists():
            raise FileNotFoundError(f"CIFAR-100-C corruption file not found: {corr_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"CIFAR-100-C labels file not found: {label_path}")
        all_images = np.load(corr_path)  # (50000, 32, 32, 3) uint8
        all_labels = np.load(label_path)  # (50000,) int
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be 1-5, got {severity}")
        start = (severity - 1) * self.SAMPLES_PER_SEVERITY
        end = start + self.SAMPLES_PER_SEVERITY
        self.images = all_images[start:end]
        self.labels = all_labels[start:end]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img = self.images[index]  # (32, 32, 3) uint8
        target = int(self.labels[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, 0, f"cifar100c_{index}"


def cifar100_transform():
    """Standard CIFAR-100 normalization transform (32x32)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])


def discover_cifar100_c_combos(data_root: str, corruption="all", severity="all") -> List[Tuple[str, int, Path]]:
    """Discover (corruption, severity, path) combos for CIFAR-100-C."""
    root = resolve_dataset_path(data_root, "cifar100-c")
    if not root.exists():
        return []
    corruptions = list(CIFAR100_C_CORRUPTIONS)
    if corruption and corruption != "all":
        wanted = [c.strip() for c in str(corruption).split(",") if c.strip()]
        corruptions = [c for c in corruptions if c in wanted]
    severities = _parse_severities(severity)
    return [(c, s, root / f"{c}.npy") for c in corruptions for s in severities]


# ---------------------------------------------------------------------------
# Non-i.i.d. / label-shift stream sampler
# ---------------------------------------------------------------------------
class LabelShiftBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that emits batches with a Dirichlet-controlled class prior.

    Simulates non-i.i.d. target streams where the class distribution drifts
    over time. At the start of each batch, a new class-proportion vector is
    sampled from a Dirichlet(alpha * uniform) distribution, and samples are
    drawn (without replacement within a class) to fill the batch according
    to that proportion.

    Args:
        labels: array of per-sample class labels.
        batch_size: number of samples per batch.
        num_batches: total number of batches to emit.
        alpha: Dirichlet concentration; small alpha => more skewed batches.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, labels, batch_size: int, num_batches: int,
                 alpha: float = 0.1, seed: int = 0):
        import numpy as np
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.num_classes = int(self.labels.max()) + 1
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist() for c in range(self.num_classes)
        }
        self.cursor = {c: 0 for c in range(self.num_classes)}

    def __iter__(self):
        for _ in range(self.num_batches):
            props = self.rng.dirichlet([self.alpha] * self.num_classes)
            counts = np.floor(props * self.batch_size).astype(int)
            counts[0] += self.batch_size - counts.sum()
            batch = []
            for c, n in enumerate(counts):
                avail = self.class_indices[c][self.cursor[c]:]
                take = min(n, len(avail))
                if take == 0:
                    continue
                batch.extend(avail[:take])
                self.cursor[c] += take
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def make_noniid_loader(
    data_root: str,
    dataset: str,
    batch_size: int = 128,
    num_workers: int = 8,
    corruption=None,
    severity=None,
    imagenet_mapping=None,
    alpha: float = 0.1,
    num_batches: int = 400,
    seed: int = 0,
    transform_override=None,
):
    """Build a DataLoader with non-i.i.d. (label-shifted) batch sampling.

    For ImageNet-C, requires the WrappedImageFolder to expose .targets.
    For CIFAR-100-C, uses the .labels array directly.
    """
    if dataset.lower() == "cifar100-c":
        if transform_override is None:
            transform_override = cifar100_transform()
        ds = CIFAR100CDataset(data_root, corruption=corruption, severity=int(severity),
                              transform=transform_override)
        labels = ds.labels
    else:
        if transform_override is None:
            transform_override = imagenet_transform()
        ds = make_single_dataset(data_root, dataset, corruption=corruption,
                                 severity=severity, transform=transform_override,
                                 imagenet_mapping=imagenet_mapping)
        if hasattr(ds, "targets"):
            labels = ds.targets
        elif hasattr(ds, "labels"):
            labels = ds.labels
        else:
            labels = [t for _, t in ds.samples]

    batch_sampler = LabelShiftBatchSampler(
        labels, batch_size=batch_size, num_batches=num_batches,
        alpha=alpha, seed=seed,
    )
    return DataLoader(
        ds, batch_sampler=batch_sampler, num_workers=num_workers,
        pin_memory=True,
    )


def get_dataset_root_for_combo(data_root: str, dataset: str, corruption=None, severity=None):
    dataset = dataset.lower()
    base = resolve_dataset_path(data_root, dataset)
    if dataset == "imagenet-c":
        if corruption is None or severity is None:
            raise ValueError("ImageNet-C requires corruption and severity")
        return base / str(corruption) / str(severity)
    return base


def make_single_dataset(
    data_root: str,
    dataset: str,
    corruption=None,
    severity=None,
    transform=None,
    imagenet_mapping: Optional[Dict[str, int]] = None,
):
    dataset = dataset.lower()
    root = get_dataset_root_for_combo(data_root, dataset, corruption, severity)
    transform = transform or imagenet_transform()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    has_class_folders = len(subdirs) > 0 and any(list_image_files(p) for p in subdirs[: min(20, len(subdirs))])
    if has_class_folders:
        class_mapping = _mapping_for_root(dataset, root, imagenet_mapping)
        return WrappedImageFolder(str(root), transform=transform, class_to_imagenet_idx=class_mapping, is_ood=0)
    return ImageListDataset(str(root), transform=transform, label=-1, is_ood=0)


def make_loader(
    data_root: str,
    dataset: str,
    batch_size: int = 64,
    num_workers: int = 8,
    corruption=None,
    severity=None,
    imagenet_mapping=None,
    shuffle: bool = False,
    max_samples: Optional[int] = None,
    transform_override=None,
):
    """Build a DataLoader. For CIFAR-100-C, uses the CIFAR100CDataset."""
    if dataset.lower() == "cifar100-c":
        from torchvision import transforms as T
        if transform_override is None:
            transform_override = cifar100_transform()
        if corruption is None or severity is None:
            raise ValueError("corruption and severity required for cifar100-c")
        ds = CIFAR100CDataset(
            data_root, corruption=corruption, severity=int(severity),
            transform=transform_override,
        )
    else:
        ds = make_single_dataset(
            data_root,
            dataset,
            corruption=corruption,
            severity=severity,
            transform=transform_override if transform_override is not None else imagenet_transform(),
            imagenet_mapping=imagenet_mapping,
        )
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _infer_mapping_status(path: Path, dataset: str, imagenet_mapping: Optional[Dict[str, int]] = None) -> str:
    if not path.exists():
        return "missing"
    dataset = dataset.lower()
    roots_to_check = []
    if dataset == "imagenet-c":
        for c in sorted([p for p in path.iterdir() if p.is_dir()]):
            sev_dirs = sorted([p for p in c.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda x: int(x.name))
            if sev_dirs:
                roots_to_check.append(sev_dirs[0])
                break
    else:
        roots_to_check.append(path)

    if not roots_to_check:
        return "no_class_folder_found"
    names = _folder_names(roots_to_check[0])
    if not names:
        return "no_class_folder_found"
    if all(n.isdigit() for n in names):
        return "numeric_folder_labels"
    if any(n.startswith("n") and len(n) >= 8 for n in names):
        mapping = imagenet_mapping or {}
        if mapping:
            ok = sum(1 for n in names if n in mapping)
            return "imagenet_synset_mapping" if ok == len(names) else f"partial_synset_mapping_{ok}_of_{len(names)}"
        fallback = _sorted_wnid_fallback_mapping(roots_to_check[0])
        if fallback:
            return f"sorted_wnid_fallback_{len(fallback)}_classes"
        return "missing_synset_mapping"
    return "sequential_folder_indices"


def scan_dataset(data_root: str, dataset: str) -> Dict[str, object]:
    dataset = dataset.lower()
    path = resolve_dataset_path(data_root, dataset)
    mapping = load_imagenet_class_mapping(data_root)
    row = {
        "dataset_name": dataset,
        "path": str(path),
        "exists": path.exists(),
        "number_of_images": 0,
        "number_of_classes": 0,
        "directory_structure": "missing",
        "severity_levels": "",
        "corruptions": "",
        "label_mapping_status": "unchecked",
        "corrupt_files_first50": 0,
        "mapping_entries_found": len(mapping),
    }
    if not path.exists():
        return row
    images = list_image_files(path)
    row["number_of_images"] = len(images)
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    row["number_of_classes"] = len(subdirs)
    row["directory_structure"] = "folder_tree"
    if dataset == "imagenet-c":
        corruptions = sorted([p.name for p in path.iterdir() if p.is_dir()])
        sevs = set()
        for c in corruptions:
            cp = path / c
            for p in cp.iterdir() if cp.exists() else []:
                if p.is_dir() and p.name.isdigit():
                    sevs.add(p.name)
        row["corruptions"] = ",".join(corruptions)
        row["severity_levels"] = ",".join(sorted(sevs, key=lambda x: int(x)))
    row["label_mapping_status"] = _infer_mapping_status(path, dataset, mapping)

    bad = 0
    for p in images[:50]:
        try:
            Image.open(p).verify()
        except Exception:
            bad += 1
    row["corrupt_files_first50"] = bad
    return row


def write_dataset_check(data_root: str, out_csv: str, out_json: str | None = None):
    rows = [scan_dataset(data_root, d) for d in ["imagenet-c", "imagenet-a", "imagenet-r", "images_largescale"]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
    return rows
