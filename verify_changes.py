"""一键验证方案C改动是否正常工作。"""
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "roco_box_detector"))

def check(cond, msg):
    print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")
    return cond

def main():
    ok = True
    print("=== 1. 模块导入 ===")
    try:
        from feature_matcher import FeatureMatcher
        from image_utils import apply_clahe, apply_preprocess, preprocess_image
        from template_cache import TemplateCache, TemplateGroup, TemplateItem, OrbFeatures, ColorHistogram
        from sequence_analyzer import (
            match_single_frame_to_patterns, vote_frame_results,
            calculate_sharpness, select_stable_frames
        )
        import json, cv2, numpy as np
        print("  [OK] All modules imported successfully")
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False

    print("\n=== 2. CLAHE 预处理 ===")
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok &= check(apply_clahe(gray).shape == gray.shape, "apply_clahe")
    ok &= check(preprocess_image(img, True, "clahe").shape == gray.shape, "preprocess_image with clahe")
    ok &= check(apply_preprocess(gray, "clahe").shape == gray.shape, "apply_preprocess clahe")

    print("\n=== 3. FeatureMatcher ===")
    fm = FeatureMatcher()
    orb = fm.extract_orb(img)
    ok &= check(len(orb.keypoints) > 0 and orb.descriptors is not None, f"extract_orb: {len(orb.keypoints)} kp")
    hist = fm.extract_color_histogram(img)
    ok &= check(hist.hsv_hist is not None, f"extract_color_histogram: shape={hist.hsv_hist.shape}")
    cs = fm.match_color_histogram(hist, hist)
    ok &= check(cs == 1.0, f"color self-match: {cs:.3f}")
    os2, _ = fm.match_orb(orb.descriptors, orb.descriptors, orb.keypoints, orb.keypoints)
    ok &= check(os2 == 1.0, f"orb self-match: {os2:.3f}")
    ens = fm.ensemble("test", 0.6, 1.0, 1.0, threshold=0.6)
    ok &= check(ens.matched and ens.ensemble_score > 0, f"ensemble: {ens.ensemble_score:.3f}")

    print("\n=== 4. 模板加载 + 特征缓存 ===")
    cfg_path = os.path.join(BASE, "roco_box_detector", "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cache = TemplateCache(cfg)
    cache.set_feature_matcher(fm)

    total_patterns = 0
    for g in cache.get_pattern_groups().values():
        total_patterns += len(g.items)
        for it in g.items:
            ok &= check(it.orb_features is not None, f"  {it.label}: {os.path.basename(it.path)} ORB cached")
    print(f"  Total pattern templates: {total_patterns}")

    total_p2 = 0
    for g in cache.get_pattern_groups_2().values():
        total_p2 += len(g.items)
        for it in g.items:
            ok &= check(it.orb_features is not None, f"  P2:{it.label}: {os.path.basename(it.path)} ORB cached")
    print(f"  Total p2 templates: {total_p2}")

    print(f"\n=== 5. 配置文件 ===")
    fm_cfg = cfg.get("feature_matching", {})
    ok &= check(fm_cfg.get("enabled"), "feature_matching.enabled = true")
    ok &= check(cfg["sequence"]["max_frames"] >= 5, f"sequence.max_frames = {cfg['sequence']['max_frames']}")
    ok &= check(cfg["sequence"]["min_vote_count"] >= 3, f"sequence.min_vote_count = {cfg['sequence']['min_vote_count']}")
    ok &= check(cfg["runtime"]["normalize_roi_width"] >= 400, f"runtime.normalize_roi_width = {cfg['runtime']['normalize_roi_width']}")
    patterns = cfg.get("patterns", {})
    all_clahe = all(p.get("preprocess_mode") == "clahe" for p in patterns.values())
    ok &= check(all_clahe, "All patterns preprocess_mode = clahe")
    patterns2 = cfg.get("patterns_2", {})
    all_clahe2 = all(p.get("preprocess_mode") == "clahe" for p in patterns2.values())
    ok &= check(all_clahe2, "All patterns_2 preprocess_mode = clahe")

    print(f"\n{'='*40}")
    print(f"总体结果: {'全部通过 [OK]' if ok else '存在失败项 [FAIL]'}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
