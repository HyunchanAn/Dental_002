import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path_cand in (ROOT_DIR, SRC_DIR):
    if path_cand not in sys.path:
        sys.path.insert(0, path_cand)

from dentex_caries import (
    CariesDetector,
    apply_clahe,
    assign_quadrant,
    map_detections_to_quadrants,
)
from dentex_caries.explain import YOLOwrapper, get_xai_heatmap
from src.data_converter import coco_to_yolo_bbox


def test_apply_clahe_preserves_shape_and_dtype():
    img = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
    out = apply_clahe(img, clip_limit=2.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_assign_and_map_quadrants():
    w, h = 2000, 1000
    assert assign_quadrant(400, 200, w, h) == 1
    assert assign_quadrant(1600, 200, w, h) == 2
    assert assign_quadrant(1600, 700, w, h) == 3
    assert assign_quadrant(400, 700, w, h) == 4

    detections = [
        {"bbox": [300, 100, 500, 300]},
        {"bbox": [1500, 600, 1700, 800]},
    ]
    mapped = map_detections_to_quadrants(detections, w, h)
    assert mapped[0]["quadrant"] == 1
    assert mapped[1]["quadrant"] == 3
    assert "UL" not in mapped[0]["quadrant_label"]


def test_coco_to_yolo_bbox_bounds():
    cx, cy, nw, nh = coco_to_yolo_bbox([100, 200, 300, 400], 1000, 1000)
    assert cx == pytest.approx(0.25)
    assert cy == pytest.approx(0.40)
    assert nw == pytest.approx(0.30)
    assert nh == pytest.approx(0.40)


def test_yolo_wrapper_unwraps_tuple():
    inner = lambda x: (x * 2.0, "aux")
    wrapper = YOLOwrapper(inner)
    t = torch.ones((1, 3, 32, 32))
    out = wrapper(t)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, t * 2.0)


def test_caries_detector_smoke_predict_filters_periapical():
    mock_box_caries = MagicMock()
    mock_box_caries.cls = [1]
    mock_box_caries.conf = [0.82]
    mock_box_caries.xyxy = [torch.tensor([10.0, 20.0, 50.0, 60.0])]

    mock_box_periapical = MagicMock()
    mock_box_periapical.cls = [2]
    mock_box_periapical.conf = [0.91]
    mock_box_periapical.xyxy = [torch.tensor([100.0, 120.0, 150.0, 160.0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box_caries, mock_box_periapical]

    with patch("dentex_caries.core.YOLO") as mock_yolo_cls:
        mock_yolo_inst = MagicMock()
        mock_yolo_inst.predict.return_value = [mock_result]
        mock_yolo_cls.return_value = mock_yolo_inst

        detector = CariesDetector(model_path="dummy.pt", conf=0.25)
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        preds, proc_img = detector.predict(img, use_clahe=True, use_sahi=False)

        assert proc_img.shape == img.shape
        assert len(preds) == 1
        assert preds[0]["name"] == "Caries"
        assert preds[0]["cls"] == 1
        assert preds[0]["conf"] == pytest.approx(0.82)
