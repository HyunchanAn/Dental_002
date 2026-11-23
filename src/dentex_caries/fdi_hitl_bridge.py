from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from dental_core.schemas.panorama_contract import (
    LesionAnnotation,
    PanoramaHiTLDocument,
    ToothAnnotation,
    validate_fdi_number,
)

CARIES_YOLO_CLASS_MAP: Dict[str, int] = {
    "Impacted": 0,
    "Caries": 1,
    "Periapical": 2,
    "Deep Caries": 3,
}


def _extract_bbox_xyxy(det: Dict[str, Any]) -> List[float]:
    raw = det.get("bbox") if det.get("bbox") is not None else det.get("box")
    if raw is None or len(raw) < 4:
        raise ValueError(f"Detection missing 4-element bbox/box: {det}")
    x1, y1, x2, y2 = map(float, raw[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _normalize_teeth_list(
    teeth_or_doc: Union[PanoramaHiTLDocument, Dict[Any, Any], Sequence[Any]]
) -> List[ToothAnnotation]:
    if isinstance(teeth_or_doc, PanoramaHiTLDocument):
        return list(teeth_or_doc.teeth)

    normalized: List[ToothAnnotation] = []
    if isinstance(teeth_or_doc, dict):
        for fdi_key, info in teeth_or_doc.items():
            try:
                fdi_int = int(fdi_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(info, dict):
                continue
            bbox = info.get("bbox_xyxy") or info.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            poly = info.get("polygon_px") or info.get("contour") or []
            normalized.append(
                ToothAnnotation(
                    fdi=fdi_int,
                    bbox_xyxy=[float(v) for v in bbox[:4]],
                    polygon_px=[[float(p[0]), float(p[1])] for p in poly],
                    confidence=float(info.get("confidence", info.get("conf", 1.0))),
                )
            )
        return normalized

    for item in teeth_or_doc:
        if isinstance(item, ToothAnnotation):
            normalized.append(item)
        elif isinstance(item, dict):
            fdi_val = int(item.get("fdi", item.get("toothNumber", 0)))
            bbox = item.get("bbox_xyxy") or item.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            poly = item.get("polygon_px") or item.get("contour") or []
            normalized.append(
                ToothAnnotation(
                    fdi=fdi_val,
                    bbox_xyxy=[float(v) for v in bbox[:4]],
                    polygon_px=[[float(p[0]), float(p[1])] for p in poly],
                    confidence=float(item.get("confidence", item.get("conf", 1.0))),
                )
            )
    return normalized


def _box_intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _point_in_polygon(pt: Tuple[float, float], polygon_px: Sequence[Sequence[float]]) -> bool:
    if len(polygon_px) < 3:
        return False
    contour = np.asarray(polygon_px, dtype=np.float32).reshape((-1, 1, 2))
    res = cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False)
    return bool(res >= 0.0)


def _is_in_crown_zone(cy: float, tooth: ToothAnnotation) -> bool:
    x1, y1, x2, y2 = tooth.bbox_xyxy
    height = max(1.0, y2 - y1)
    quadrant = int(tooth.fdi) // 10 if tooth.fdi else 0
    is_maxillary = quadrant in (1, 2, 5, 6)
    if is_maxillary:
        # Maxillary crown occupies the inferior 65% of the tooth box
        return cy >= (y1 + 0.35 * height)
    # Mandibular crown occupies the superior 65% of the tooth box
    return cy <= (y1 + 0.65 * height)


def map_caries_to_008_teeth(
    detections: Sequence[Dict[str, Any]],
    teeth_or_doc: Union[PanoramaHiTLDocument, Dict[Any, Any], Sequence[Any]],
    min_ioa: float = 0.15,
    definite_conf: float = 0.45,
    exclude_periapical: bool = True,
) -> List[Dict[str, Any]]:
    """
    Map Dental_002 caries detections onto Dental_008 FDI tooth instances using
    polygon containment, Intersection-over-Lesion-Area (IoA), and anatomical crown-zone weighting.
    """
    teeth = _normalize_teeth_list(teeth_or_doc)
    mapped_results: List[Dict[str, Any]] = []

    for det in detections:
        label = str(det.get("name") or det.get("label") or "Caries")
        if exclude_periapical and "periapical" in label.lower():
            continue

        bbox = _extract_bbox_xyxy(det)
        conf = float(det.get("conf", det.get("confidence", 0.5)))
        lesion_w = max(1.0, bbox[2] - bbox[0])
        lesion_h = max(1.0, bbox[3] - bbox[1])
        lesion_area = lesion_w * lesion_h
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])

        best_tooth: Optional[ToothAnnotation] = None
        best_score = 0.0
        best_ioa = 0.0
        best_in_poly = False

        for tooth in teeth:
            if not validate_fdi_number(tooth.fdi, allow_unknown=False):
                continue
            inter = _box_intersection_area(bbox, tooth.bbox_xyxy)
            ioa = inter / lesion_area
            poly = tooth.ensure_polygon()
            in_poly = _point_in_polygon((cx, cy), poly)
            if ioa < min_ioa and not in_poly:
                continue

            crown_bonus = 0.15 if _is_in_crown_zone(cy, tooth) else 0.0
            poly_bonus = 0.35 if in_poly else 0.0
            score = ioa + poly_bonus + crown_bonus
            if score > best_score:
                best_score = score
                best_tooth = tooth
                best_ioa = ioa
                best_in_poly = in_poly

        matched_fdi = int(best_tooth.fdi) if best_tooth is not None else None
        is_definite = (
            matched_fdi is not None
            and conf >= definite_conf
            and (best_in_poly or best_ioa >= 0.35)
        )
        clinical_status = "DEFINITE" if is_definite else "POSSIBLE"

        enriched = dict(det)
        enriched["bbox"] = [round(v, 2) for v in bbox]
        enriched["label"] = label
        enriched["confidence"] = round(conf, 4)
        enriched["fdi"] = matched_fdi
        enriched["ioa"] = round(best_ioa, 4)
        enriched["in_tooth_polygon"] = best_in_poly
        enriched["clinical_status"] = clinical_status
        mapped_results.append(enriched)

    if isinstance(teeth_or_doc, PanoramaHiTLDocument):
        preserved = [
            lesion
            for lesion in teeth_or_doc.lesions
            if lesion.category != "caries" or lesion.clinician_corrected
        ]
        existing_boxes = [
            lesion.bbox_xyxy
            for lesion in preserved
            if lesion.category == "caries"
        ]
        for item in mapped_results:
            is_dup = any(
                _box_intersection_area(item["bbox"], eb)
                / max(1.0, (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]))
                >= 0.6
                for eb in existing_boxes
            )
            if is_dup:
                continue
            preserved.append(
                LesionAnnotation(
                    category="caries",
                    label=item["label"],
                    bbox_xyxy=item["bbox"],
                    fdi=item["fdi"],
                    confidence=item["confidence"],
                    clinical_status=item["clinical_status"],
                    clinician_corrected=False,
                )
            )
        teeth_or_doc.lesions = preserved

    return mapped_results


def _clip_yolo_norm_box(
    bbox_xyxy: Sequence[float], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    if width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(float(width), float(bbox_xyxy[0])))
    y1 = max(0.0, min(float(height), float(bbox_xyxy[1])))
    x2 = max(0.0, min(float(width), float(bbox_xyxy[2])))
    y2 = max(0.0, min(float(height), float(bbox_xyxy[3])))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    cx = (x1 + 0.5 * bw) / float(width)
    cy = (y1 + 0.5 * bh) / float(height)
    nw = bw / float(width)
    nh = bh / float(height)
    return (round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6))


def export_hitl_caries_to_yolo(
    golden_json_paths: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    mode: str = "full",
    images_dir: Optional[Union[str, Path]] = None,
    patch_pad_ratio: float = 0.12,
    only_verified_golden: bool = False,
) -> Dict[str, Any]:
    """
    Export clinician-reviewed Dental_015 HiTL Golden JSON files into YOLO detection format
    for Dental_002 retraining (`mode='full'` panorama or `mode='tooth_patch'` per-tooth ROI).
    """
    if mode not in ("full", "tooth_patch"):
        raise ValueError(f"Unsupported export mode: {mode}. Expected 'full' or 'tooth_patch'.")

    out_root = Path(output_dir)
    labels_dir = out_root / "labels" / "train"
    out_images_dir = out_root / "images" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if images_dir is not None:
        out_images_dir.mkdir(parents=True, exist_ok=True)

    exported_files = 0
    exported_boxes = 0

    for json_path in golden_json_paths:
        doc = PanoramaHiTLDocument.load_json(json_path)
        if only_verified_golden and doc.hitl_status != "verified_golden":
            continue

        caries_lesions = [l for l in doc.lesions if l.category == "caries"]
        stem = Path(doc.image_meta.filename or Path(json_path).stem).stem
        img_w, img_h = doc.image_meta.width, doc.image_meta.height

        src_img: Optional[np.ndarray] = None
        if images_dir is not None and doc.image_meta.filename:
            cand = Path(images_dir) / doc.image_meta.filename
            if cand.exists():
                src_img = cv2.imread(str(cand))

        if mode == "full":
            lines: List[str] = []
            for lesion in caries_lesions:
                cls_id = CARIES_YOLO_CLASS_MAP.get(lesion.label, 1)
                norm_tuple = _clip_yolo_norm_box(lesion.bbox_xyxy, img_w, img_h)
                if norm_tuple is None:
                    continue
                lines.append(
                    f"{cls_id} {norm_tuple[0]:.6f} {norm_tuple[1]:.6f} {norm_tuple[2]:.6f} {norm_tuple[3]:.6f}"
                )
                exported_boxes += 1

            label_file = labels_dir / f"{stem}.txt"
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if src_img is not None:
                cv2.imwrite(str(out_images_dir / f"{stem}.png"), src_img)
            exported_files += 1
        else:
            teeth_by_fdi = {int(t.fdi): t for t in doc.teeth if validate_fdi_number(t.fdi)}
            for fdi, tooth in teeth_by_fdi.items():
                tx1, ty1, tx2, ty2 = tooth.bbox_xyxy
                pad_x = (tx2 - tx1) * patch_pad_ratio
                pad_y = (ty2 - ty1) * patch_pad_ratio
                px1 = int(max(0, round(tx1 - pad_x)))
                py1 = int(max(0, round(ty1 - pad_y)))
                px2 = int(min(img_w, round(tx2 + pad_x)))
                py2 = int(min(img_h, round(ty2 + pad_y)))
                pw = px2 - px1
                ph = py2 - py1
                if pw <= 4 or ph <= 4:
                    continue

                tooth_lesions = [
                    l
                    for l in caries_lesions
                    if (l.fdi == fdi) or (_box_intersection_area(l.bbox_xyxy, [px1, py1, px2, py2]) > 0)
                ]
                if not tooth_lesions:
                    continue

                patch_lines: List[str] = []
                for lesion in tooth_lesions:
                    cls_id = CARIES_YOLO_CLASS_MAP.get(lesion.label, 1)
                    shifted = [
                        lesion.bbox_xyxy[0] - px1,
                        lesion.bbox_xyxy[1] - py1,
                        lesion.bbox_xyxy[2] - px1,
                        lesion.bbox_xyxy[3] - py1,
                    ]
                    norm_tuple = _clip_yolo_norm_box(shifted, pw, ph)
                    if norm_tuple is None:
                        continue
                    patch_lines.append(
                        f"{cls_id} {norm_tuple[0]:.6f} {norm_tuple[1]:.6f} {norm_tuple[2]:.6f} {norm_tuple[3]:.6f}"
                    )
                    exported_boxes += 1

                if not patch_lines:
                    continue
                patch_stem = f"{stem}_fdi{fdi}"
                (labels_dir / f"{patch_stem}.txt").write_text(
                    "\n".join(patch_lines) + "\n", encoding="utf-8"
                )
                if src_img is not None:
                    crop = src_img[py1:py2, px1:px2]
                    if crop.size > 0:
                        cv2.imwrite(str(out_images_dir / f"{patch_stem}.png"), crop)
                exported_files += 1

    data_yaml_content = (
        "path: .\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: Impacted\n"
        "  1: Caries\n"
        "  2: Periapical\n"
        "  3: Deep Caries\n"
    )
    data_yaml_path = out_root / "data.yaml"
    data_yaml_path.write_text(data_yaml_content, encoding="utf-8")

    return {
        "mode": mode,
        "output_dir": str(out_root),
        "data_yaml": str(data_yaml_path),
        "exported_files": exported_files,
        "exported_boxes": exported_boxes,
    }
