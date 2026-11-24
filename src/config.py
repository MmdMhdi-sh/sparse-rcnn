"""Paths, dataset settings, model sizes and training hyperparameters."""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ("circle", "square", "triangle")    # class ids 0, 1, 2; background is id 3


@dataclass
class DataConfig:
    img_size: int = 128
    n_pool: int = 1000            # assignment: 1000 images, split 900 train / 100 validation
    n_val: int = 100
    n_test: int = 1000            # separate held-out set, generated from its own seed
    pool_seed: int = 0            # the dataset is fixed across training seeds
    test_seed: int = 1
    max_shapes: int = 3           # 1 to 3 shapes per image
    iou_threshold: float = 0.1    # max IoU between two ground-truth boxes
    visibility_threshold: float = 0.75   # min fraction of every box left uncovered
    min_color: int = 64           # avoid near-black shapes on the black background
    batch_size: int = 16


@dataclass
class ModelConfig:
    num_classes: int = 3
    num_proposals: int = 20
    feature_dim: int = 128
    num_stages: int = 3
    roi_size: int = 7
    use_fpn: bool = True          # False: a single 16x16 feature map, as in the assignment text
    # FPN level by normalised box area (shapes cover at most ~0.15 of the image)
    level_area_thresholds: tuple = (0.1, 0.3)
    max_log_scale: float = 5.0    # clamp on the predicted log width/height change


@dataclass
class LossConfig:
    cost_class: float = 1.0       # Hungarian matching cost weights
    cost_bbox: float = 1.0
    weight_cls: float = 1.0       # loss weights
    weight_bbox: float = 2.0
    focal_gamma: float = 2.0


@dataclass
class TrainConfig:
    lr: float = 5e-5              # Adam, as in the assignment
    lr_step: int = 25             # StepLR every 25 epochs, gamma 0.1 (as in the course version)
    lr_gamma: float = 0.1
    grad_clip: float = 10.0
    max_epochs: int = 80
    patience: int = 20            # early stopping on validation mAP
    seed: int = 42
    max_steps_per_epoch: int = 0  # 0 = full epoch (smoke tests use a small number)


@dataclass
class ExperimentConfig:
    name: str
    label: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


EXPERIMENTS = {
    "sparse_rcnn": ExperimentConfig(
        name="sparse_rcnn",
        label="Sparse R-CNN (FPN P2-P4, 3 stages)"),
    "single_scale": ExperimentConfig(
        name="single_scale",
        label="Single 16x16 feature map (no FPN)",
        model=ModelConfig(use_fpn=False)),
    # Not in the default run: 5x more synthetic training images, same validation and test sets.
    "more_data": ExperimentConfig(
        name="more_data",
        label="Sparse R-CNN, 4,900 training images",
        data=DataConfig(n_pool=5000)),
}
DEFAULT_EXPERIMENTS = ("sparse_rcnn", "single_scale")


def with_seed(exp: ExperimentConfig, seed: int) -> ExperimentConfig:
    return replace(exp, train=replace(exp.train, seed=seed))
