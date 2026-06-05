"""Multi-strategy feature matching: ORB keypoints, color histograms, ensemble scoring.

Provides fallback/verification methods when pure template matching is uncertain.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EnsembleScore:
    label: str
    template_score: float
    color_score: float
    orb_score: float
    ensemble_score: float
    matched: bool = False


@dataclass
class OrbFeatures:
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]


@dataclass
class ColorHistogram:
    hsv_hist: np.ndarray
    mean_h: float
    mean_s: float
    mean_v: float


class FeatureMatcher:
    """Multi-strategy feature matcher with ensemble scoring."""

    def __init__(
        self,
        orb_nfeatures: int = 200,
        orb_scale_factor: float = 1.2,
        orb_nlevels: int = 4,
        color_bins_h: int = 30,
        color_bins_s: int = 32,
        color_bins_v: int = 32,
        ensemble_weights: Optional[Tuple[float, float, float]] = None,
    ):
        self.orb = cv2.ORB_create(
            nfeatures=orb_nfeatures,
            scaleFactor=orb_scale_factor,
            nlevels=orb_nlevels,
            edgeThreshold=7,
            patchSize=31,
            fastThreshold=20,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.color_bins = (color_bins_h, color_bins_s, color_bins_v)
        self.ensemble_weights = ensemble_weights or (0.5, 0.3, 0.2)

    def extract_orb(self, image: np.ndarray) -> OrbFeatures:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kp, des = self.orb.detectAndCompute(gray, None)
        return OrbFeatures(keypoints=kp or [], descriptors=des)

    def extract_color_histogram(self, image: np.ndarray) -> ColorHistogram:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.color_bins[0], self.color_bins[1], self.color_bins[2]],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        return ColorHistogram(hsv_hist=hist, mean_h=mean_h, mean_s=mean_s, mean_v=mean_v)

    def match_orb(
        self,
        des1: Optional[np.ndarray],
        des2: Optional[np.ndarray],
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        ratio_threshold: float = 0.75,
        min_matches: int = 3,
    ) -> Tuple[float, int]:
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0
        try:
            matches = self.bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            return 0.0, 0

        good = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance < ratio_threshold * n.distance:
                    good.append(m)

        if len(good) < min_matches:
            return float(len(good)) / max(len(kp1), 1) * 0.3, len(good)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if mask is None:
            return float(min(len(good), min_matches * 2)) / max(len(kp1), 1) * 0.5, len(good)

        inliers = int(mask.sum())
        total = max(len(kp1), 1)
        score = min(1.0, (inliers / total) * 1.5)
        return score, inliers

    def match_color_histogram(
        self, hist1: ColorHistogram, hist2: ColorHistogram
    ) -> float:
        cmp = cv2.compareHist(hist1.hsv_hist, hist2.hsv_hist, cv2.HISTCMP_BHATTACHARYYA)
        hist_sim = max(0.0, 1.0 - cmp)

        h_diff = min(abs(hist1.mean_h - hist2.mean_h) / 90.0, 1.0)
        s_diff = min(abs(hist1.mean_s - hist2.mean_s) / 255.0, 1.0)
        v_diff = min(abs(hist1.mean_v - hist2.mean_v) / 255.0, 1.0)
        mean_diff = (h_diff * 0.5 + s_diff * 0.3 + v_diff * 0.2)
        mean_sim = max(0.0, 1.0 - mean_diff)

        return 0.6 * hist_sim + 0.4 * mean_sim

    def ensemble(
        self,
        label: str,
        template_score: float,
        color_score: float,
        orb_score: float,
        threshold: float = 0.65,
    ) -> EnsembleScore:
        w_t, w_c, w_o = self.ensemble_weights
        w_total = w_t + w_c + w_o

        effective_template = template_score
        effective_color = color_score if color_score >= 0.3 else 0.0
        effective_orb = orb_score if orb_score >= 0.15 else 0.0

        ensemble_score = (
            effective_template * w_t
            + effective_color * w_c
            + effective_orb * w_o
        ) / w_total

        matched = ensemble_score >= threshold or (
            template_score >= threshold * 1.1
        )

        return EnsembleScore(
            label=label,
            template_score=template_score,
            color_score=color_score,
            orb_score=orb_score,
            ensemble_score=ensemble_score,
            matched=matched,
        )
