from typing import List
from pydantic import BaseModel


class Args(BaseModel):
    """
    # ------------------------ Model ------------------------
    model,
    # ------------------------ DATA ------------------------
    source,
    imgsz,
    batch,
    data_yaml,
    val,
    name,
    cache,

    # ------------------------ TRAIN ------------------------
    weights,
    epochs,
    workers,
    freeze,
    # ------------------------ VAL ------------------------
    task,
    # ------------------------ DETECT ------------------------
    save,
    conf,
    iou,
    augment,
    half,
    nosave,
    classes,
    # ------------------------ EXPORT ------------------------
    format,
    imgsz,
    half,
    """

    model: str
    source: str
    imgsz: int
    batch: int
    data_yaml: str
    val: str
    name: str
    cache: str
    weights: str
    epochs: int
    workers: int
    freeze: str
    task: str
    save: str
    conf: float
    iou: float
    augment: bool
    half: bool
    nosave: bool
    classes: str
    format: str

    @property
    def vllm_args(self):
        return [
            "--model",
            self.model,
            "--source",
            self.source,
            "--imgsz",
            self.imgsz,
            "--batch",
            self.batch,
            "--data_yaml",
            self.data_yaml,
            "--val",
            self.val,
            "--name",
            self.name,
            "--cache",
            self.cache,
            "--weights",
            self.weights,
            "--epochs",
            self.epochs,
            "--workers",
            self.workers,
            "--freeze",
            self.freeze,
            "--task",
            self.task,
            "--save",
            self.save,
            "--conf",
            self.conf,
            "--iou",
            self.iou,
            "--augment",
            self.augment,
            "--half",
            self.half,
            "--nosave",
            self.nosave,
            "--classes",
            self.classes,
            "--format",
            self.format,
        ]

    @property
    def llama_cpp_args(self):
        return [
            "--model",
            self.model,
            "--source",
            self.source,
            "--imgsz",
            self.imgsz,
            "--batch",
            self.batch,
            "--data_yaml",
            self.data_yaml,
            "--val",
            self.val,
            "--name",
            self.name,
            "--cache",
            self.cache,
            "--weights",
            self.weights,
            "--epochs",
            self.epochs,
            "--workers",
            self.workers,
            "--freeze",
            self.freeze,
            "--task",
            self.task,
            "--save",
            self.save,
            "--conf",
            self.conf,
            "--iou",
            self.iou,
            "--augment",
            self.augment,
            "--half",
            self.half,
            "--nosave",
            self.nosave,
            "--classes",
            self.classes,
            "--format",
            self.format,
        ]
