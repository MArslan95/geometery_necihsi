"""
Incremental Learning Dataset Manager
====================================

Strict non-exemplar HSI class-incremental dataset manager.

Critical label policy
---------------------
This manager assumes labels passed into it are already training labels in a
sequential class space 0..K-1.

That means:
- For datasets with background in the raw GT, the loader must remove background
  and remap foreground labels to 0..K-1.
- For datasets where raw class 0 is a real class, the loader must preserve it.
  Class 0 must NOT be treated as background.

The manager does not delete class 0. It treats whatever labels it receives as
real classes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HSIPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray):
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if len(patches) != len(labels):
            raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        return self.patches[idx], self.labels[idx]


class IncrementalHSIDataset:
    """
    Strict non-exemplar incremental dataset manager.

    Important:
    The dataset manager never assumes label 0 is background. If label 0 is in
    labels, it is treated as a valid class.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        gt_shape: Tuple[int, int],
        GT: np.ndarray,
        base_classes: int,
        increment: int,
        train_ratio: float = 0.2,
        val_ratio: float = 0.1,
        seed: int = 42,
        shuffle_order: bool = False,
        device: str = "cuda",
        min_train_per_class: int = 20,
        num_workers: int = 0,
        strict_non_exemplar: bool = True,
        target_names: Optional[List[str]] = None,
        label_policy: Optional[Dict] = None,
    ):
        # Raw arrays from ImageCubes.
        self.patches = np.asarray(patches, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.gt_shape = gt_shape
        self.GT = GT

        if len(self.patches) != len(self.labels):
            raise ValueError(f"patch/label length mismatch: {len(self.patches)} vs {len(self.labels)}")
        if len(self.coords) != len(self.labels):
            raise ValueError(f"coord/label length mismatch: {len(self.coords)} vs {len(self.labels)}")
        if self.labels.size == 0:
            raise ValueError("Empty labels passed to IncrementalHSIDataset.")
        if self.labels.min() < 0:
            raise ValueError(
                f"Negative labels passed to IncrementalHSIDataset: min={self.labels.min()}. "
                f"Loader must remove background/ignore labels before incremental split."
            )

        # Settings.
        self.base_classes = int(base_classes)
        self.increment = int(increment)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.device = str(device)
        self.min_train_per_class = int(min_train_per_class)
        self.num_workers = int(num_workers)
        self.strict_non_exemplar = bool(strict_non_exemplar)
        self.target_names = target_names
        self.label_policy = label_policy or {}

        self.pin_memory = self.device.startswith("cuda")

        # Cache for semantic tokens / concept tokens.
        self._semantic_token_cache: Dict[Tuple[int, str], torch.Tensor] = {}

        # Protocol state.
        self.current_phase: int = 0
        self.finalized_phases: Set[int] = set()
        self.finalized_classes: Set[int] = set()
        self._memory_build_active: bool = False
        self._memory_build_classes: Set[int] = set()

        # Class order and remapping.
        # labels may already be 0..K-1, but we still remap to sequential IDs
        # according to class_order. If label 0 exists, it is included.
        self.all_classes = sorted(int(x) for x in np.unique(self.labels).tolist())
        if self.all_classes[0] != 0:
            print(
                f"[IncrementalHSIDataset:WARN] smallest label is {self.all_classes[0]}, "
                f"not 0. The manager will remap to sequential IDs."
            )

        self.num_classes = len(self.all_classes)

        if self.base_classes <= 0 or self.base_classes > self.num_classes:
            raise ValueError(f"base_classes={self.base_classes} invalid for num_classes={self.num_classes}")
        if self.increment <= 0:
            raise ValueError(f"increment must be > 0, got {self.increment}")

        if shuffle_order:
            rng = np.random.RandomState(self.seed)
            self.class_order = rng.permutation(self.all_classes).tolist()
        else:
            self.class_order = list(self.all_classes)

        self.label_map = {global_id: seq_id for seq_id, global_id in enumerate(self.class_order)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.remapped_labels = np.array(
            [self.label_map[int(l)] for l in self.labels],
            dtype=np.int64,
        )

        # target_names provided by the loader are indexed by input label id
        # after ImageCubes mapping. Convert them once to sequential phase ids.
        self.target_names_by_seq = self._build_target_names_by_seq(target_names)

        self._validate_class_zero_policy()
        self._validate_full_class_coverage()

        # Phase partition in sequential label space.
        remaining_classes = self.num_classes - self.base_classes
        self.num_phases = 1 + int(np.ceil(remaining_classes / self.increment))

        self.phase_to_classes: Dict[int, List[int]] = {}
        self.class_to_phase: Dict[int, int] = {}

        for phase in range(self.num_phases):
            if phase == 0:
                cls_list = list(range(self.base_classes))
            else:
                start = self.base_classes + (phase - 1) * self.increment
                end = min(start + self.increment, self.num_classes)
                cls_list = list(range(start, end))

            self.phase_to_classes[phase] = cls_list
            for cls in cls_list:
                self.class_to_phase[int(cls)] = int(phase)

        self._create_splits()
        self._print_stats()

    # ============================================================
    # Validation
    # ============================================================
    def _validate_class_zero_policy(self) -> None:
        raw_has_zero = 0 in set(int(x) for x in self.labels.tolist())
        remap_has_zero = 0 in set(int(x) for x in self.remapped_labels.tolist())

        if raw_has_zero and not remap_has_zero:
            raise RuntimeError(
                "Class 0 existed in input labels but disappeared after remapping. "
                "This must never happen for class-0-real datasets."
            )

        if self.label_policy and not bool(self.label_policy.get("has_background", True)):
            if 0 not in self.label_policy.get("raw_class_values", []):
                print(
                    "[IncrementalHSIDataset:WARN] label_policy says no background, "
                    "but raw_class_values does not contain 0. Check dataset metadata."
                )

    def _build_target_names_by_seq(self, target_names: Optional[List[str]]) -> Optional[List[str]]:
        if target_names is None:
            return None

        out: List[str] = []
        for sid in range(len(self.all_classes)):
            gid = self.inv_label_map.get(sid, sid)
            if int(gid) < len(target_names):
                out.append(str(target_names[int(gid)]))
            else:
                out.append(f"Class {gid}")
        return out

    def _validate_full_class_coverage(self) -> None:
        present = set(int(x) for x in np.unique(self.remapped_labels).tolist())
        expected = set(range(self.num_classes))
        missing = sorted(expected - present)
        extra = sorted(present - expected)

        if missing or extra:
            raise RuntimeError(
                f"Remapped label space is broken. Missing={missing}, extra={extra}, "
                f"num_classes={self.num_classes}, class_order={self.class_order}"
            )

    def _validate_phase(self, phase: int) -> int:
        phase = int(phase)
        if phase < 0 or phase >= self.num_phases:
            raise ValueError(f"Invalid phase {phase}. Valid range: 0..{self.num_phases - 1}")
        return phase

    # ============================================================
    # Basic mapping helpers
    # ============================================================
    def seq_to_global(self, seq_id: int) -> int:
        return int(self.inv_label_map[int(seq_id)])

    def global_to_seq(self, global_id: int) -> int:
        return int(self.label_map[int(global_id)])

    def get_phase_classes(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        return list(self.phase_to_classes[phase])

    def get_classes_up_to_phase(self, phase: int) -> List[int]:
        phase = self._validate_phase(phase)
        classes: List[int] = []
        for p in range(phase + 1):
            classes.extend(self.phase_to_classes[p])
        return classes

    # ============================================================
    # Protocol controls
    # ============================================================
    def start_phase(self, phase: int) -> None:
        phase = self._validate_phase(phase)
        self.current_phase = phase
        self._invalidate_train_caches_for_locked_classes()

    def finalize_phase(self, phase: int) -> None:
        phase = self._validate_phase(phase)

        self.finalized_phases.add(phase)
        self.finalized_classes.update(self.phase_to_classes[phase])
        self._memory_build_active = False
        self._memory_build_classes.clear()
        self._invalidate_train_caches_for_locked_classes()

    def is_phase_finalized(self, phase: int) -> bool:
        return int(phase) in self.finalized_phases

    def is_class_finalized(self, cls: int) -> bool:
        return int(cls) in self.finalized_classes

    def get_accessible_train_classes(self) -> List[int]:
        if not self.strict_non_exemplar:
            return list(range(self.num_classes))
        return list(self.phase_to_classes[self.current_phase])

    def _invalidate_train_caches_for_locked_classes(self) -> None:
        keys_to_delete = []
        accessible = set(self.get_accessible_train_classes())

        for key in list(self._semantic_token_cache.keys()):
            cls, descriptor = key
            if "_train" not in descriptor:
                continue
            if self.strict_non_exemplar and int(cls) not in accessible:
                keys_to_delete.append(key)

        for k in keys_to_delete:
            del self._semantic_token_cache[k]

    def _is_train_access_allowed(self, cls: int) -> bool:
        cls = int(cls)

        if not self.strict_non_exemplar:
            return True

        if cls in self.phase_to_classes[self.current_phase]:
            return True

        if self._memory_build_active and cls in self._memory_build_classes:
            return True

        return False

    def _check_class_split_access(self, cls: int, split: str) -> None:
        cls = int(cls)
        split = str(split).lower()

        if split != "train":
            return

        if not self._is_train_access_allowed(cls):
            phase_of_cls = self.class_to_phase.get(cls, None)
            raise PermissionError(
                f"Strict non-exemplar protocol violation: raw TRAIN access denied for class {cls} "
                f"(phase={phase_of_cls}, current_phase={self.current_phase}, finalized={cls in self.finalized_classes}). "
                f"Use stored geometry memory instead of old raw training patches."
            )

    @contextmanager
    def memory_build_context(self, phase: int):
        phase = int(phase)
        if phase != self.current_phase:
            raise ValueError(
                f"memory_build_context phase={phase} must match current_phase={self.current_phase}"
            )

        allowed_classes = set(self.phase_to_classes[phase])

        prev_active = self._memory_build_active
        prev_classes = set(self._memory_build_classes)

        self._memory_build_active = True
        self._memory_build_classes = allowed_classes

        try:
            yield
        finally:
            self._memory_build_active = prev_active
            self._memory_build_classes = prev_classes

    # ============================================================
    # Split creation
    # ============================================================
    def _create_splits(self) -> None:
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []

        for seq_id in range(self.num_classes):
            class_indices = np.where(self.remapped_labels == seq_id)[0]
            n_samples = len(class_indices)

            if n_samples == 0:
                continue

            rng = np.random.RandomState(self.seed + seq_id)
            shuffled = rng.permutation(class_indices)

            if n_samples == 1:
                self.train_indices.extend(shuffled.tolist())
                continue

            if n_samples == 2:
                self.train_indices.append(int(shuffled[0]))
                self.test_indices.append(int(shuffled[1]))
                continue

            n_train = max(self.min_train_per_class, int(round(n_samples * self.train_ratio)))
            n_val = max(1, int(round(n_samples * self.val_ratio)))

            if n_train + n_val >= n_samples:
                overflow = (n_train + n_val) - (n_samples - 1)

                reduce_val = min(overflow, max(0, n_val - 1))
                n_val -= reduce_val
                overflow -= reduce_val

                if overflow > 0:
                    n_train = max(1, n_train - overflow)

            n_train = max(1, min(n_train, n_samples - 2))
            n_val = max(1, min(n_val, n_samples - n_train - 1))
            n_test = n_samples - n_train - n_val

            assert n_train >= 1
            assert n_val >= 1
            assert n_test >= 1
            assert n_train + n_val + n_test == n_samples

            self.train_indices.extend(shuffled[:n_train].tolist())
            self.val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
            self.test_indices.extend(shuffled[n_train + n_val:].tolist())

        self.train_indices = np.array(self.train_indices, dtype=np.int64)
        self.val_indices = np.array(self.val_indices, dtype=np.int64)
        self.test_indices = np.array(self.test_indices, dtype=np.int64)

    # ============================================================
    # Diagnostics
    # ============================================================
    def _print_stats(self) -> None:
        print("[IncrementalHSIDataset] Initialized")
        print(f"  Total classes: {self.num_classes} | Phases: {self.num_phases}")
        print(f"  Input classes: {self.all_classes}")
        print(f"  Class order: {self.class_order}")
        print(f"  Strict non-exemplar: {self.strict_non_exemplar}")
        print(f"  Label 0 present as class: {0 in self.all_classes}")

        for p in range(self.num_phases):
            global_ids = [self.seq_to_global(sid) for sid in self.phase_to_classes[p]]
            names = []
            if self.target_names_by_seq is not None:
                for sid in self.phase_to_classes[p]:
                    if int(sid) < len(self.target_names_by_seq):
                        names.append(self.target_names_by_seq[int(sid)])
                    else:
                        names.append(f"Class {self.seq_to_global(sid)}")
            print(f"  Phase {p}: Sequential {self.phase_to_classes[p]} (Input labels {global_ids})")
            if names:
                print(f"           Names: {names}")

    def get_class_counts(self) -> Dict[int, int]:
        return {
            int(c): int((self.remapped_labels == int(c)).sum())
            for c in range(self.num_classes)
        }

    def get_split_class_counts(self, split: str = "train") -> Dict[int, int]:
        indices = self._get_split_indices(split)
        return {
            int(c): int((self.remapped_labels[indices] == int(c)).sum())
            for c in range(self.num_classes)
        }

    def protocol_state(self) -> Dict[str, object]:
        return {
            "current_phase": int(self.current_phase),
            "strict_non_exemplar": bool(self.strict_non_exemplar),
            "finalized_phases": sorted(int(p) for p in self.finalized_phases),
            "finalized_classes": sorted(int(c) for c in self.finalized_classes),
            "accessible_train_classes": self.get_accessible_train_classes(),
            "memory_build_active": bool(self._memory_build_active),
            "memory_build_classes": sorted(int(c) for c in self._memory_build_classes),
        }

    def assert_no_old_train_access(self, cls: int) -> None:
        self._check_class_split_access(int(cls), "train")

    def _make_loader(self, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            raise ValueError(
                "Requested DataLoader has zero samples. Check phase split, class order, "
                "or strict non-exemplar access policy."
            )

        dataset = HSIPatchDataset(self.patches[idx], self.remapped_labels[idx])
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=False,
        )

    # ============================================================
    # Split access helpers
    # ============================================================
    def _get_split_indices(self, split: str) -> np.ndarray:
        split = split.lower()
        if split == "train":
            return self.train_indices
        if split == "val":
            return self.val_indices
        if split == "test":
            return self.test_indices
        if split == "all":
            return np.arange(len(self.remapped_labels), dtype=np.int64)
        raise ValueError(f"Unknown split '{split}'. Use one of: train, val, test, all")

    def get_class_indices(self, cls: int, split: str = "train") -> np.ndarray:
        cls = int(cls)
        split = split.lower()
        self._check_class_split_access(cls, split)

        indices = self._get_split_indices(split)
        mask = self.remapped_labels[indices] == cls
        return indices[mask]

    def get_class_patches(self, cls: int, split: str = "train") -> np.ndarray:
        idx = self.get_class_indices(cls, split=split)
        if len(idx) == 0:
            raise ValueError(f"No samples found for class {cls} in split '{split}'")
        return self.patches[idx]

    # ============================================================
    # DataLoaders
    # ============================================================
    def get_phase_dataloader(
        self,
        phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        phase = self._validate_phase(phase)
        split = split.lower()

        active_classes = self.phase_to_classes[phase]

        if split == "train":
            if self.strict_non_exemplar and phase != self.current_phase:
                raise PermissionError(
                    f"Raw TRAIN loader requested for phase {phase}, but current_phase is {self.current_phase}. "
                    "Strict non-exemplar mode only allows current-phase raw train access."
                )

        indices = self._get_split_indices(split)
        mask = np.isin(self.remapped_labels[indices], active_classes)
        idx = indices[mask]

        return self._make_loader(
            idx,
            batch_size=batch_size,
            shuffle=(shuffle if split == "train" else False),
        )

    def get_cumulative_dataloader(
        self,
        up_to_phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
        allow_train_old: bool = False,
    ) -> DataLoader:
        up_to_phase = self._validate_phase(up_to_phase)
        split = split.lower()

        if split == "train" and self.strict_non_exemplar and not allow_train_old:
            # Critical protocol rule: training loaders never expose old raw samples.
            active_classes = list(self.phase_to_classes[self.current_phase])
        else:
            active_classes: List[int] = []
            for p in range(up_to_phase + 1):
                active_classes.extend(self.phase_to_classes[p])

        indices = self._get_split_indices(split)
        mask = np.isin(self.remapped_labels[indices], active_classes)
        idx = indices[mask]

        return self._make_loader(
            idx,
            batch_size=batch_size,
            shuffle=(shuffle if split == "train" else False),
        )

    def get_cumulative_test_data(self, phase: int):
        phase = self._validate_phase(phase)
        active_classes = self.get_classes_up_to_phase(phase)
        mask = np.isin(self.remapped_labels[self.test_indices], active_classes)
        idx = self.test_indices[mask]
        return self.patches[idx], self.remapped_labels[idx], self.coords[idx]

    # ============================================================
    # K-means concept extraction
    # ============================================================
    def _kmeans_numpy(
        self,
        x: np.ndarray,
        k: int,
        seed: int = 42,
        max_iters: int = 30,
    ) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got shape={x.shape}")

        n, d = x.shape
        if n == 0:
            raise ValueError("Empty input to k-means")

        k = max(1, min(int(k), n))
        rng = np.random.RandomState(seed)

        centers = np.empty((k, d), dtype=np.float32)
        first = rng.randint(0, n)
        centers[0] = x[first]
        dist2 = ((x - centers[0]) ** 2).sum(axis=1)

        for i in range(1, k):
            probs = dist2 / max(dist2.sum(), 1e-12)
            idx = rng.choice(n, p=probs)
            centers[i] = x[idx]
            dist2 = np.minimum(dist2, ((x - centers[i]) ** 2).sum(axis=1))

        for _ in range(max_iters):
            assign = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            new_centers = centers.copy()
            for i in range(k):
                mask = assign == i
                if mask.any():
                    new_centers[i] = x[mask].mean(axis=0)

            if np.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        return centers.astype(np.float32)

    # ============================================================
    # Semantic concept / token access
    # ============================================================
    def get_class_concept_tokens(
        self,
        cls: int,
        split: str = "train",
        num_concepts: int = 4,
        use_cache: bool = True,
    ) -> torch.Tensor:
        cls = int(cls)
        split = split.lower()
        num_concepts = int(max(1, num_concepts))

        self._check_class_split_access(cls, split)

        cache_key = (cls, f"concept_{split}_{num_concepts}")
        if use_cache and cache_key in self._semantic_token_cache:
            return self._semantic_token_cache[cache_key].clone()

        class_indices = self.get_class_indices(cls, split=split)
        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {cls} in split '{split}'")

        class_patches = self.patches[class_indices]
        per_sample_summaries = class_patches.mean(axis=(2, 3)).astype(np.float32)

        if per_sample_summaries.shape[0] <= num_concepts:
            concepts = per_sample_summaries
        else:
            concepts = self._kmeans_numpy(
                per_sample_summaries,
                k=num_concepts,
                seed=self.seed + cls,
            )

        token = torch.from_numpy(concepts).float()

        if use_cache:
            self._semantic_token_cache[cache_key] = token.clone()

        return token

    def get_class_semantic_token(
        self,
        cls: int,
        split: str = "train",
        use_cache: bool = True,
    ) -> torch.Tensor:
        cls = int(cls)
        split = split.lower()
        self._check_class_split_access(cls, split)

        cache_key = (cls, f"coarse_{split}")
        if use_cache and cache_key in self._semantic_token_cache:
            return self._semantic_token_cache[cache_key].clone()

        concepts = self.get_class_concept_tokens(
            cls=cls,
            split=split,
            num_concepts=4,
            use_cache=use_cache,
        )
        token = concepts.mean(dim=0, keepdim=True)

        if use_cache:
            self._semantic_token_cache[cache_key] = token.clone()

        return token

    def clear_semantic_token_cache(self) -> None:
        self._semantic_token_cache.clear()








# """
# Incremental Learning Dataset Manager
# ====================================

# Strict non-exemplar HSI class-incremental dataset manager.

# Critical label policy
# ---------------------
# This manager assumes labels passed into it are already training labels in a
# sequential class space 0..K-1.

# That means:
# - For datasets with background in the raw GT, the loader must remove background
#   and remap foreground labels to 0..K-1.
# - For datasets where raw class 0 is a real class, the loader must preserve it.
#   Class 0 must NOT be treated as background.

# The manager does not delete class 0. It treats whatever labels it receives as
# real classes.
# """

# from __future__ import annotations

# from contextlib import contextmanager
# from typing import Dict, List, Optional, Set, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset


# class HSIPatchDataset(Dataset):
#     def __init__(self, patches: np.ndarray, labels: np.ndarray):
#         patches = np.ascontiguousarray(patches, dtype=np.float32)
#         labels = np.asarray(labels, dtype=np.int64).reshape(-1)

#         if len(patches) != len(labels):
#             raise ValueError(f"patch/label length mismatch: {len(patches)} vs {len(labels)}")

#         self.patches = torch.from_numpy(patches).float()
#         self.labels = torch.from_numpy(labels).long()

#     def __len__(self) -> int:
#         return int(len(self.labels))

#     def __getitem__(self, idx: int):
#         return self.patches[idx], self.labels[idx]


# class IncrementalHSIDataset:
#     """
#     Strict non-exemplar incremental dataset manager.

#     Important:
#     The dataset manager never assumes label 0 is background. If label 0 is in
#     labels, it is treated as a valid class.
#     """

#     def __init__(
#         self,
#         patches: np.ndarray,
#         labels: np.ndarray,
#         coords: np.ndarray,
#         gt_shape: Tuple[int, int],
#         GT: np.ndarray,
#         base_classes: int,
#         increment: int,
#         train_ratio: float = 0.2,
#         val_ratio: float = 0.1,
#         seed: int = 42,
#         shuffle_order: bool = False,
#         device: str = "cuda",
#         min_train_per_class: int = 20,
#         num_workers: int = 0,
#         strict_non_exemplar: bool = True,
#         target_names: Optional[List[str]] = None,
#         label_policy: Optional[Dict] = None,
#     ):
#         # Raw arrays from ImageCubes.
#         self.patches = np.asarray(patches, dtype=np.float32)
#         self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
#         self.coords = np.asarray(coords, dtype=np.int64)
#         self.gt_shape = gt_shape
#         self.GT = GT

#         if len(self.patches) != len(self.labels):
#             raise ValueError(f"patch/label length mismatch: {len(self.patches)} vs {len(self.labels)}")
#         if len(self.coords) != len(self.labels):
#             raise ValueError(f"coord/label length mismatch: {len(self.coords)} vs {len(self.labels)}")
#         if self.labels.size == 0:
#             raise ValueError("Empty labels passed to IncrementalHSIDataset.")
#         if self.labels.min() < 0:
#             raise ValueError(
#                 f"Negative labels passed to IncrementalHSIDataset: min={self.labels.min()}. "
#                 f"Loader must remove background/ignore labels before incremental split."
#             )

#         # Settings.
#         self.base_classes = int(base_classes)
#         self.increment = int(increment)
#         self.train_ratio = float(train_ratio)
#         self.val_ratio = float(val_ratio)
#         self.seed = int(seed)
#         self.device = str(device)
#         self.min_train_per_class = int(min_train_per_class)
#         self.num_workers = int(num_workers)
#         self.strict_non_exemplar = bool(strict_non_exemplar)
#         self.target_names = target_names
#         self.label_policy = label_policy or {}

#         self.pin_memory = self.device.startswith("cuda")

#         # Cache for semantic tokens / concept tokens.
#         self._semantic_token_cache: Dict[Tuple[int, str], torch.Tensor] = {}

#         # Protocol state.
#         self.current_phase: int = 0
#         self.finalized_phases: Set[int] = set()
#         self.finalized_classes: Set[int] = set()
#         self._memory_build_active: bool = False
#         self._memory_build_classes: Set[int] = set()

#         # Class order and remapping.
#         # labels may already be 0..K-1, but we still remap to sequential IDs
#         # according to class_order. If label 0 exists, it is included.
#         self.all_classes = sorted(int(x) for x in np.unique(self.labels).tolist())
#         if self.all_classes[0] != 0:
#             print(
#                 f"[IncrementalHSIDataset:WARN] smallest label is {self.all_classes[0]}, "
#                 f"not 0. The manager will remap to sequential IDs."
#             )

#         self.num_classes = len(self.all_classes)

#         if self.base_classes <= 0 or self.base_classes > self.num_classes:
#             raise ValueError(f"base_classes={self.base_classes} invalid for num_classes={self.num_classes}")
#         if self.increment <= 0:
#             raise ValueError(f"increment must be > 0, got {self.increment}")

#         if shuffle_order:
#             rng = np.random.RandomState(self.seed)
#             self.class_order = rng.permutation(self.all_classes).tolist()
#         else:
#             self.class_order = list(self.all_classes)

#         self.label_map = {global_id: seq_id for seq_id, global_id in enumerate(self.class_order)}
#         self.inv_label_map = {v: k for k, v in self.label_map.items()}

#         self.remapped_labels = np.array(
#             [self.label_map[int(l)] for l in self.labels],
#             dtype=np.int64,
#         )

#         # target_names provided by the loader are indexed by input label id
#         # after ImageCubes mapping. Convert them once to sequential phase ids.
#         self.target_names_by_seq = self._build_target_names_by_seq(target_names)

#         self._validate_class_zero_policy()
#         self._validate_full_class_coverage()

#         # Phase partition in sequential label space.
#         remaining_classes = self.num_classes - self.base_classes
#         self.num_phases = 1 + int(np.ceil(remaining_classes / self.increment))

#         self.phase_to_classes: Dict[int, List[int]] = {}
#         self.class_to_phase: Dict[int, int] = {}

#         for phase in range(self.num_phases):
#             if phase == 0:
#                 cls_list = list(range(self.base_classes))
#             else:
#                 start = self.base_classes + (phase - 1) * self.increment
#                 end = min(start + self.increment, self.num_classes)
#                 cls_list = list(range(start, end))

#             self.phase_to_classes[phase] = cls_list
#             for cls in cls_list:
#                 self.class_to_phase[int(cls)] = int(phase)

#         self._create_splits()
#         self._print_stats()

#     # ============================================================
#     # Validation
#     # ============================================================
#     def _validate_class_zero_policy(self) -> None:
#         raw_has_zero = 0 in set(int(x) for x in self.labels.tolist())
#         remap_has_zero = 0 in set(int(x) for x in self.remapped_labels.tolist())

#         if raw_has_zero and not remap_has_zero:
#             raise RuntimeError(
#                 "Class 0 existed in input labels but disappeared after remapping. "
#                 "This must never happen for class-0-real datasets."
#             )

#         if self.label_policy and not bool(self.label_policy.get("has_background", True)):
#             if 0 not in self.label_policy.get("raw_class_values", []):
#                 print(
#                     "[IncrementalHSIDataset:WARN] label_policy says no background, "
#                     "but raw_class_values does not contain 0. Check dataset metadata."
#                 )

#     def _build_target_names_by_seq(self, target_names: Optional[List[str]]) -> Optional[List[str]]:
#         if target_names is None:
#             return None

#         out: List[str] = []
#         for sid in range(len(self.all_classes)):
#             gid = self.inv_label_map.get(sid, sid)
#             if int(gid) < len(target_names):
#                 out.append(str(target_names[int(gid)]))
#             else:
#                 out.append(f"Class {gid}")
#         return out

#     def _validate_full_class_coverage(self) -> None:
#         present = set(int(x) for x in np.unique(self.remapped_labels).tolist())
#         expected = set(range(self.num_classes))
#         missing = sorted(expected - present)
#         extra = sorted(present - expected)

#         if missing or extra:
#             raise RuntimeError(
#                 f"Remapped label space is broken. Missing={missing}, extra={extra}, "
#                 f"num_classes={self.num_classes}, class_order={self.class_order}"
#             )

#     def _validate_phase(self, phase: int) -> int:
#         phase = int(phase)
#         if phase < 0 or phase >= self.num_phases:
#             raise ValueError(f"Invalid phase {phase}. Valid range: 0..{self.num_phases - 1}")
#         return phase

#     # ============================================================
#     # Basic mapping helpers
#     # ============================================================
#     def seq_to_global(self, seq_id: int) -> int:
#         return int(self.inv_label_map[int(seq_id)])

#     def global_to_seq(self, global_id: int) -> int:
#         return int(self.label_map[int(global_id)])

#     def get_phase_classes(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         return list(self.phase_to_classes[phase])

#     def get_classes_up_to_phase(self, phase: int) -> List[int]:
#         phase = self._validate_phase(phase)
#         classes: List[int] = []
#         for p in range(phase + 1):
#             classes.extend(self.phase_to_classes[p])
#         return classes

#     # ============================================================
#     # Protocol controls
#     # ============================================================
#     def start_phase(self, phase: int) -> None:
#         phase = self._validate_phase(phase)
#         self.current_phase = phase
#         self._invalidate_train_caches_for_locked_classes()

#     def finalize_phase(self, phase: int) -> None:
#         phase = self._validate_phase(phase)

#         self.finalized_phases.add(phase)
#         self.finalized_classes.update(self.phase_to_classes[phase])
#         self._memory_build_active = False
#         self._memory_build_classes.clear()
#         self._invalidate_train_caches_for_locked_classes()

#     def is_phase_finalized(self, phase: int) -> bool:
#         return int(phase) in self.finalized_phases

#     def is_class_finalized(self, cls: int) -> bool:
#         return int(cls) in self.finalized_classes

#     def get_accessible_train_classes(self) -> List[int]:
#         if not self.strict_non_exemplar:
#             return list(range(self.num_classes))
#         return list(self.phase_to_classes[self.current_phase])

#     def _invalidate_train_caches_for_locked_classes(self) -> None:
#         keys_to_delete = []
#         accessible = set(self.get_accessible_train_classes())

#         for key in list(self._semantic_token_cache.keys()):
#             cls, descriptor = key
#             if "_train" not in descriptor:
#                 continue
#             if self.strict_non_exemplar and int(cls) not in accessible:
#                 keys_to_delete.append(key)

#         for k in keys_to_delete:
#             del self._semantic_token_cache[k]

#     def _is_train_access_allowed(self, cls: int) -> bool:
#         cls = int(cls)

#         if not self.strict_non_exemplar:
#             return True

#         if cls in self.phase_to_classes[self.current_phase]:
#             return True

#         if self._memory_build_active and cls in self._memory_build_classes:
#             return True

#         return False

#     def _check_class_split_access(self, cls: int, split: str) -> None:
#         cls = int(cls)
#         split = str(split).lower()

#         if split != "train":
#             return

#         if not self._is_train_access_allowed(cls):
#             phase_of_cls = self.class_to_phase.get(cls, None)
#             raise PermissionError(
#                 f"Strict non-exemplar protocol violation: raw TRAIN access denied for class {cls} "
#                 f"(phase={phase_of_cls}, current_phase={self.current_phase}, finalized={cls in self.finalized_classes}). "
#                 f"Use stored geometry memory instead of old raw training patches."
#             )

#     @contextmanager
#     def memory_build_context(self, phase: int):
#         phase = int(phase)
#         if phase != self.current_phase:
#             raise ValueError(
#                 f"memory_build_context phase={phase} must match current_phase={self.current_phase}"
#             )

#         allowed_classes = set(self.phase_to_classes[phase])

#         prev_active = self._memory_build_active
#         prev_classes = set(self._memory_build_classes)

#         self._memory_build_active = True
#         self._memory_build_classes = allowed_classes

#         try:
#             yield
#         finally:
#             self._memory_build_active = prev_active
#             self._memory_build_classes = prev_classes

#     # ============================================================
#     # Split creation
#     # ============================================================
#     def _create_splits(self) -> None:
#         self.train_indices: List[int] = []
#         self.val_indices: List[int] = []
#         self.test_indices: List[int] = []

#         for seq_id in range(self.num_classes):
#             class_indices = np.where(self.remapped_labels == seq_id)[0]
#             n_samples = len(class_indices)

#             if n_samples == 0:
#                 continue

#             rng = np.random.RandomState(self.seed + seq_id)
#             shuffled = rng.permutation(class_indices)

#             if n_samples == 1:
#                 self.train_indices.extend(shuffled.tolist())
#                 continue

#             if n_samples == 2:
#                 self.train_indices.append(int(shuffled[0]))
#                 self.test_indices.append(int(shuffled[1]))
#                 continue

#             n_train = max(self.min_train_per_class, int(round(n_samples * self.train_ratio)))
#             n_val = max(1, int(round(n_samples * self.val_ratio)))

#             if n_train + n_val >= n_samples:
#                 overflow = (n_train + n_val) - (n_samples - 1)

#                 reduce_val = min(overflow, max(0, n_val - 1))
#                 n_val -= reduce_val
#                 overflow -= reduce_val

#                 if overflow > 0:
#                     n_train = max(1, n_train - overflow)

#             n_train = max(1, min(n_train, n_samples - 2))
#             n_val = max(1, min(n_val, n_samples - n_train - 1))
#             n_test = n_samples - n_train - n_val

#             assert n_train >= 1
#             assert n_val >= 1
#             assert n_test >= 1
#             assert n_train + n_val + n_test == n_samples

#             self.train_indices.extend(shuffled[:n_train].tolist())
#             self.val_indices.extend(shuffled[n_train:n_train + n_val].tolist())
#             self.test_indices.extend(shuffled[n_train + n_val:].tolist())

#         self.train_indices = np.array(self.train_indices, dtype=np.int64)
#         self.val_indices = np.array(self.val_indices, dtype=np.int64)
#         self.test_indices = np.array(self.test_indices, dtype=np.int64)

#     # ============================================================
#     # Diagnostics
#     # ============================================================
#     def _print_stats(self) -> None:
#         print("[IncrementalHSIDataset] Initialized")
#         print(f"  Total classes: {self.num_classes} | Phases: {self.num_phases}")
#         print(f"  Input classes: {self.all_classes}")
#         print(f"  Class order: {self.class_order}")
#         print(f"  Strict non-exemplar: {self.strict_non_exemplar}")
#         print(f"  Label 0 present as class: {0 in self.all_classes}")

#         for p in range(self.num_phases):
#             global_ids = [self.seq_to_global(sid) for sid in self.phase_to_classes[p]]
#             names = []
#             if self.target_names_by_seq is not None:
#                 for sid in self.phase_to_classes[p]:
#                     if int(sid) < len(self.target_names_by_seq):
#                         names.append(self.target_names_by_seq[int(sid)])
#                     else:
#                         names.append(f"Class {self.seq_to_global(sid)}")
#             print(f"  Phase {p}: Sequential {self.phase_to_classes[p]} (Input labels {global_ids})")
#             if names:
#                 print(f"           Names: {names}")

#     def get_class_counts(self) -> Dict[int, int]:
#         return {
#             int(c): int((self.remapped_labels == int(c)).sum())
#             for c in range(self.num_classes)
#         }

#     def get_split_class_counts(self, split: str = "train") -> Dict[int, int]:
#         indices = self._get_split_indices(split)
#         return {
#             int(c): int((self.remapped_labels[indices] == int(c)).sum())
#             for c in range(self.num_classes)
#         }

#     def protocol_state(self) -> Dict[str, object]:
#         return {
#             "current_phase": int(self.current_phase),
#             "strict_non_exemplar": bool(self.strict_non_exemplar),
#             "finalized_phases": sorted(int(p) for p in self.finalized_phases),
#             "finalized_classes": sorted(int(c) for c in self.finalized_classes),
#             "accessible_train_classes": self.get_accessible_train_classes(),
#             "memory_build_active": bool(self._memory_build_active),
#             "memory_build_classes": sorted(int(c) for c in self._memory_build_classes),
#         }

#     def assert_no_old_train_access(self, cls: int) -> None:
#         self._check_class_split_access(int(cls), "train")

#     def _make_loader(self, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
#         idx = np.asarray(idx, dtype=np.int64)
#         if idx.size == 0:
#             raise ValueError(
#                 "Requested DataLoader has zero samples. Check phase split, class order, "
#                 "or strict non-exemplar access policy."
#             )

#         dataset = HSIPatchDataset(self.patches[idx], self.remapped_labels[idx])
#         return DataLoader(
#             dataset,
#             batch_size=int(batch_size),
#             shuffle=bool(shuffle),
#             pin_memory=self.pin_memory,
#             num_workers=self.num_workers,
#             drop_last=False,
#         )

#     # ============================================================
#     # Split access helpers
#     # ============================================================
#     def _get_split_indices(self, split: str) -> np.ndarray:
#         split = split.lower()
#         if split == "train":
#             return self.train_indices
#         if split == "val":
#             return self.val_indices
#         if split == "test":
#             return self.test_indices
#         if split == "all":
#             return np.arange(len(self.remapped_labels), dtype=np.int64)
#         raise ValueError(f"Unknown split '{split}'. Use one of: train, val, test, all")

#     def get_class_indices(self, cls: int, split: str = "train") -> np.ndarray:
#         cls = int(cls)
#         split = split.lower()
#         self._check_class_split_access(cls, split)

#         indices = self._get_split_indices(split)
#         mask = self.remapped_labels[indices] == cls
#         return indices[mask]

#     def get_class_patches(self, cls: int, split: str = "train") -> np.ndarray:
#         idx = self.get_class_indices(cls, split=split)
#         if len(idx) == 0:
#             raise ValueError(f"No samples found for class {cls} in split '{split}'")
#         return self.patches[idx]

#     # ============================================================
#     # DataLoaders
#     # ============================================================
#     def get_phase_dataloader(
#         self,
#         phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#     ) -> DataLoader:
#         phase = self._validate_phase(phase)
#         split = split.lower()

#         active_classes = self.phase_to_classes[phase]

#         if split == "train":
#             if self.strict_non_exemplar and phase != self.current_phase:
#                 raise PermissionError(
#                     f"Raw TRAIN loader requested for phase {phase}, but current_phase is {self.current_phase}. "
#                     "Strict non-exemplar mode only allows current-phase raw train access."
#                 )

#         indices = self._get_split_indices(split)
#         mask = np.isin(self.remapped_labels[indices], active_classes)
#         idx = indices[mask]

#         return self._make_loader(
#             idx,
#             batch_size=batch_size,
#             shuffle=(shuffle if split == "train" else False),
#         )

#     def get_cumulative_dataloader(
#         self,
#         up_to_phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#         allow_train_old: bool = False,
#     ) -> DataLoader:
#         up_to_phase = self._validate_phase(up_to_phase)
#         split = split.lower()

#         if split == "train" and self.strict_non_exemplar and not allow_train_old:
#             # Critical protocol rule: training loaders never expose old raw samples.
#             active_classes = list(self.phase_to_classes[self.current_phase])
#         else:
#             active_classes: List[int] = []
#             for p in range(up_to_phase + 1):
#                 active_classes.extend(self.phase_to_classes[p])

#         indices = self._get_split_indices(split)
#         mask = np.isin(self.remapped_labels[indices], active_classes)
#         idx = indices[mask]

#         return self._make_loader(
#             idx,
#             batch_size=batch_size,
#             shuffle=(shuffle if split == "train" else False),
#         )

#     def get_cumulative_test_data(self, phase: int):
#         phase = self._validate_phase(phase)
#         active_classes = self.get_classes_up_to_phase(phase)
#         mask = np.isin(self.remapped_labels[self.test_indices], active_classes)
#         idx = self.test_indices[mask]
#         return self.patches[idx], self.remapped_labels[idx], self.coords[idx]

#     # ============================================================
#     # K-means concept extraction
#     # ============================================================
#     def _kmeans_numpy(
#         self,
#         x: np.ndarray,
#         k: int,
#         seed: int = 42,
#         max_iters: int = 30,
#     ) -> np.ndarray:
#         if x.ndim != 2:
#             raise ValueError(f"Expected x to be 2D, got shape={x.shape}")

#         n, d = x.shape
#         if n == 0:
#             raise ValueError("Empty input to k-means")

#         k = max(1, min(int(k), n))
#         rng = np.random.RandomState(seed)

#         centers = np.empty((k, d), dtype=np.float32)
#         first = rng.randint(0, n)
#         centers[0] = x[first]
#         dist2 = ((x - centers[0]) ** 2).sum(axis=1)

#         for i in range(1, k):
#             probs = dist2 / max(dist2.sum(), 1e-12)
#             idx = rng.choice(n, p=probs)
#             centers[i] = x[idx]
#             dist2 = np.minimum(dist2, ((x - centers[i]) ** 2).sum(axis=1))

#         for _ in range(max_iters):
#             assign = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
#             new_centers = centers.copy()
#             for i in range(k):
#                 mask = assign == i
#                 if mask.any():
#                     new_centers[i] = x[mask].mean(axis=0)

#             if np.allclose(new_centers, centers, atol=1e-5):
#                 centers = new_centers
#                 break
#             centers = new_centers

#         return centers.astype(np.float32)

#     # ============================================================
#     # Semantic concept / token access
#     # ============================================================
#     def get_class_concept_tokens(
#         self,
#         cls: int,
#         split: str = "train",
#         num_concepts: int = 4,
#         use_cache: bool = True,
#     ) -> torch.Tensor:
#         cls = int(cls)
#         split = split.lower()
#         num_concepts = int(max(1, num_concepts))

#         self._check_class_split_access(cls, split)

#         cache_key = (cls, f"concept_{split}_{num_concepts}")
#         if use_cache and cache_key in self._semantic_token_cache:
#             return self._semantic_token_cache[cache_key].clone()

#         class_indices = self.get_class_indices(cls, split=split)
#         if len(class_indices) == 0:
#             raise ValueError(f"No samples found for class {cls} in split '{split}'")

#         class_patches = self.patches[class_indices]
#         per_sample_summaries = class_patches.mean(axis=(2, 3)).astype(np.float32)

#         if per_sample_summaries.shape[0] <= num_concepts:
#             concepts = per_sample_summaries
#         else:
#             concepts = self._kmeans_numpy(
#                 per_sample_summaries,
#                 k=num_concepts,
#                 seed=self.seed + cls,
#             )

#         token = torch.from_numpy(concepts).float()

#         if use_cache:
#             self._semantic_token_cache[cache_key] = token.clone()

#         return token

#     def get_class_semantic_token(
#         self,
#         cls: int,
#         split: str = "train",
#         use_cache: bool = True,
#     ) -> torch.Tensor:
#         cls = int(cls)
#         split = split.lower()
#         self._check_class_split_access(cls, split)

#         cache_key = (cls, f"coarse_{split}")
#         if use_cache and cache_key in self._semantic_token_cache:
#             return self._semantic_token_cache[cache_key].clone()

#         concepts = self.get_class_concept_tokens(
#             cls=cls,
#             split=split,
#             num_concepts=4,
#             use_cache=use_cache,
#         )
#         token = concepts.mean(dim=0, keepdim=True)

#         if use_cache:
#             self._semantic_token_cache[cache_key] = token.clone()

#         return token

#     def clear_semantic_token_cache(self) -> None:
#         self._semantic_token_cache.clear()
