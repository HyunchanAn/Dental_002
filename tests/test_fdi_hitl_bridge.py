import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path_cand in (ROOT_DIR, SRC_DIR):
    if path_cand not in sys.path:
        sys.path.insert(0, path_cand)

from dental_core.schemas.panorama_contract import (
    LesionAnnotation,
    PanoramaHiTLDocument,
    PanoramaImageMeta,
    ToothAnnotation,
)
from dentex_caries import export_hitl_caries_to_yolo, map_caries_to_008_teeth


def _build_sample_doc() -> PanoramaHiTLDocument:
    return PanoramaHiTLDocument(
        image_meta=PanoramaImageMeta(filename="pano_sample.png", width=1000, height=500),
        teeth=[
            ToothAnnotation(
                fdi=16,
                bbox_xyxy=[200.0, 100.0, 280.0, 220.0],
                polygon_px=[[205.0, 105.0], [275.0, 105.0], [275.0, 215.0], [205.0, 215.0]],
            ),
            ToothAnnotation(
                fdi=36,
                bbox_xyxy=[700.0, 280.0, 780.0, 410.0],
                polygon_px=[[705.0, 285.0], [775.0, 285.0], [775.0, 405.0], [705.0, 405.0]],
            ),
        ],
    )


def test_map_caries_to_008_teeth_and_doc_merge():
    doc = _build_sample_doc()
    # Pre-populate a clinician-corrected caries box on #36 that must be preserved
    doc.lesions.append(
        LesionAnnotation(
            category="caries",
            label="Deep Caries",
            bbox_xyxy=[720.0, 290.0, 755.0, 325.0],
            fdi=36,
            confidence=1.0,
            clinical_status="DEFINITE",
            clinician_corrected=True,
        )
    )

    detections = [
        # Crown caries on maxillary right molar #16 (inferior half of box)
        {"box": [220.0, 175.0, 255.0, 210.0], "conf": 0.78, "name": "Caries"},
        # Duplicate overlap with clinician-corrected #36 box (should not duplicate in doc.lesions)
        {"box": [722.0, 292.0, 753.0, 323.0], "conf": 0.65, "name": "Caries"},
        # Periapical class must be excluded for Dental_012 domain separation
        {"box": [220.0, 80.0, 260.0, 110.0], "conf": 0.90, "name": "Periapical"},
        # Unmatched background box
        {"box": [20.0, 20.0, 45.0, 45.0], "conf": 0.40, "name": "Caries"},
    ]

    mapped = map_caries_to_008_teeth(detections, doc)
    assert len(mapped) == 3
    assert mapped[0]["fdi"] == 16
    assert mapped[0]["in_tooth_polygon"] is True
    assert mapped[0]["clinical_status"] == "DEFINITE"

    assert mapped[1]["fdi"] == 36
    assert mapped[2]["fdi"] is None
    assert mapped[2]["clinical_status"] == "POSSIBLE"

    # doc.lesions should contain preserved clinician_corrected #36 + new #16 + unmatched background (3 total)
    assert len(doc.lesions) == 3
    corrected = [l for l in doc.lesions if l.clinician_corrected]
    assert len(corrected) == 1
    assert corrected[0].fdi == 36
    assert corrected[0].label == "Deep Caries"


def test_map_caries_accepts_legacy_008_dict_format():
    legacy_008 = {
        46: {
            "bbox": [150.0, 300.0, 230.0, 430.0],
            "contour": [[152.0, 302.0], [228.0, 302.0], [228.0, 428.0], [152.0, 428.0]],
            "conf": 0.92,
        }
    }
    dets = [{"bbox": [175.0, 310.0, 205.0, 340.0], "confidence": 0.81, "label": "Deep Caries"}]
    mapped = map_caries_to_008_teeth(dets, legacy_008)
    assert len(mapped) == 1
    assert mapped[0]["fdi"] == 46
    assert mapped[0]["clinical_status"] == "DEFINITE"


def test_export_hitl_caries_to_yolo_full_and_patch(tmp_path: Path):
    doc = _build_sample_doc()
    doc.hitl_status = "verified_golden"
    doc.lesions = [
        LesionAnnotation(
            category="caries",
            label="Caries",
            bbox_xyxy=[220.0, 170.0, 260.0, 210.0],
            fdi=16,
            confidence=0.95,
            clinician_corrected=True,
        ),
        LesionAnnotation(
            category="caries",
            label="Deep Caries",
            bbox_xyxy=[715.0, 290.0, 760.0, 330.0],
            fdi=36,
            confidence=1.0,
            clinician_corrected=True,
        ),
    ]

    img_dir = tmp_path / "raw_images"
    img_dir.mkdir()
    dummy_img = np.full((500, 1000, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "pano_sample.png"), dummy_img)

    golden_json = tmp_path / "golden_001.json"
    doc.save_json(golden_json)

    full_out = tmp_path / "yolo_full"
    res_full = export_hitl_caries_to_yolo(
        [golden_json], full_out, mode="full", images_dir=img_dir, only_verified_golden=True
    )
    assert res_full["exported_files"] == 1
    assert res_full["exported_boxes"] == 2
    lbl_lines = (full_out / "labels" / "train" / "pano_sample.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lbl_lines) == 2
    assert lbl_lines[0].startswith("1 0.240000 0.380000 0.040000 0.080000")
    assert lbl_lines[1].startswith("3 ")

    patch_out = tmp_path / "yolo_patch"
    res_patch = export_hitl_caries_to_yolo(
        [golden_json], patch_out, mode="tooth_patch", images_dir=img_dir, only_verified_golden=True
    )
    assert res_patch["exported_files"] == 2
    assert res_patch["exported_boxes"] == 2
    assert (patch_out / "labels" / "train" / "pano_sample_fdi16.txt").exists()
    assert (patch_out / "images" / "train" / "pano_sample_fdi36.png").exists()
