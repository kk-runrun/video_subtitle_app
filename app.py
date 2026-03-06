import os
import re
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import Dict, List, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st

try:
    import enchant
except Exception:
    enchant = None

st.set_page_config(page_title="视频字幕提取工作站", layout="wide")
st.title("视频字幕自动提取工作站")
st.caption("OpenCV + EasyOCR | 输出: 视频名称 / 视频原字幕")


@st.cache_resource(show_spinner=False)
def get_ocr_reader(use_gpu: bool) -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=use_gpu)


@st.cache_resource(show_spinner=False)
def get_en_dict():
    if enchant is None:
        return None
    try:
        return enchant.Dict("en_US")
    except Exception:
        return None


def ensure_video_path(path_text: str) -> str:
    path_text = path_text.strip().strip('"')
    return os.path.abspath(path_text) if path_text else ""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_output_text(text: str) -> str:
    text = text.replace("_", " ")
    text = clean_text(text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def valid_text(text: str, min_core_chars: int) -> bool:
    core = re.sub(r"[^A-Za-z0-9]+", "", text)
    return len(core) >= min_core_chars


def preprocess_gray_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    gray = cv2.equalizeHist(gray)
    return gray


def preprocess_white_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    bw = cv2.resize(bw, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    return bw


def preprocess_white_roi_alt(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        29,
        9,
    )
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = cv2.resize(bw, None, fx=1.20, fy=1.20, interpolation=cv2.INTER_CUBIC)
    return bw


def group_to_lines(
    detections: List[Tuple[List[List[float]], str, float]],
    roi_h: int,
    min_core_chars: int,
    conf_threshold: float,
    box_conf_floor: float,
) -> List[Tuple[str, float]]:
    if not detections:
        return []

    boxes: List[Dict[str, float]] = []
    for bbox, raw_text, conf in detections:
        text = clean_text(raw_text)
        if not text:
            continue
        if float(conf) < box_conf_floor:
            continue
        if not valid_text(text, min_core_chars):
            continue

        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        h_box = max(1.0, y2 - y1)
        if h_box < max(8.0, roi_h * 0.040):
            continue

        boxes.append(
            {
                "text": text,
                "x1": x1,
                "x2": x2,
                "y": (y1 + y2) / 2.0,
                "conf": float(conf),
            }
        )

    if not boxes:
        return []

    boxes.sort(key=lambda b: (b["y"], b["x1"]))
    line_tol = max(10.0, roi_h * 0.038)

    lines: List[Dict[str, object]] = []
    for b in boxes:
        hit = None
        for line in lines:
            if abs(b["y"] - line["y_mean"]) <= line_tol:
                hit = line
                break
        if hit is None:
            lines.append({"y_mean": b["y"], "items": [b]})
        else:
            items = hit["items"]
            items.append(b)
            hit["y_mean"] = sum(i["y"] for i in items) / len(items)

    lines.sort(key=lambda line: line["y_mean"])
    out: List[Tuple[str, float]] = []
    for line in lines:
        items = sorted(line["items"], key=lambda i: i["x1"])
        merged = normalize_output_text(" ".join(item["text"] for item in items))
        if not valid_text(merged, min_core_chars):
            continue

        avg_conf = float(sum(item["conf"] for item in items) / len(items))
        if avg_conf < conf_threshold:
            continue
        out.append((merged, avg_conf))

    return out


def extract_lines_from_roi(
    reader: easyocr.Reader,
    roi_bgr: np.ndarray,
    mode: str,
    min_core_chars: int,
    conf_threshold: float,
) -> List[Tuple[str, float]]:
    roi_h = roi_bgr.shape[0]

    if mode == "gray":
        img = preprocess_gray_roi(roi_bgr)
        detections = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.60,
            low_text=0.20,
            link_threshold=0.30,
            width_ths=0.80,
            height_ths=0.80,
        )
        box_floor = max(0.24, min(conf_threshold - 0.18, 0.72))
        return group_to_lines(detections, roi_h, min_core_chars, conf_threshold, box_floor)

    img1 = preprocess_white_roi(roi_bgr)

    # 白字独立阈值：在全局0.75时适度放宽，减少漏首字母
    white_conf_threshold = max(0.52, min(0.72, conf_threshold - 0.10))
    white_min_core_chars = max(3, min_core_chars)

    det1 = reader.readtext(
        img1,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.45,
        low_text=0.10,
        link_threshold=0.18,
        width_ths=0.95,
        height_ths=0.95,
    )
    img2 = preprocess_white_roi_alt(roi_bgr)
    det2 = reader.readtext(
        img2,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.40,
        low_text=0.08,
        link_threshold=0.16,
        width_ths=0.98,
        height_ths=0.98,
    )

    merged = list(det1) + list(det2)
    box_floor = max(0.18, min(white_conf_threshold - 0.20, 0.66))
    return group_to_lines(merged, roi_h, white_min_core_chars, white_conf_threshold, box_floor)


def get_target_rois(frame: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    h, w = frame.shape[:2]
    y_gray = int(h * 0.78)
    y_upper = int(h * 0.60)
    y_upper_bottom = int(h * 0.82)
    x_mid = int(w * 0.5)
    overlap = int(w * 0.08)

    rois = [
        (frame[y_gray:h, 0:w], "gray"),
        (frame[y_upper:y_upper_bottom, 0:min(w, x_mid + overlap)], "white"),
        (frame[y_upper:y_upper_bottom, max(0, x_mid - overlap):w], "white"),
    ]
    return [(roi, mode) for roi, mode in rois if roi.size > 0]


def text_similarity(a: str, b: str) -> float:
    a_norm = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
    b_norm = re.sub(r"[^a-z0-9]+", " ", b.lower()).strip()
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def mergeable(a: str, b: str) -> bool:
    a_key = re.sub(r"\s+", " ", a).strip().lower()
    b_key = re.sub(r"\s+", " ", b).strip().lower()
    if a_key == b_key:
        return True

    a_core = re.sub(r"[^a-z0-9]+", "", a_key)
    b_core = re.sub(r"[^a-z0-9]+", "", b_key)
    if min(len(a_core), len(b_core)) < 8:
        return False

    sim = text_similarity(a, b)
    if sim >= 0.86:
        return True

    short, long_ = (a_core, b_core) if len(a_core) <= len(b_core) else (b_core, a_core)
    return short in long_ and sim >= 0.72


def choose_best_suggestion(token: str, suggestions: List[str]) -> str:
    if not suggestions:
        return token
    t_low = token.lower()
    best = token
    best_score = 0.0
    for cand in suggestions[:8]:
        cand_low = cand.lower()
        if not cand_low.isalpha():
            continue
        if abs(len(cand_low) - len(t_low)) > 2:
            continue
        sim = SequenceMatcher(None, t_low, cand_low).ratio()
        if sim > best_score:
            best_score = sim
            best = cand
    return best if best_score >= 0.72 else token


def one_edit_valid_words(token: str, en_dict) -> List[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = set()
    n = len(token)

    # replace
    for i in range(n):
        for ch in letters:
            if ch == token[i]:
                continue
            cand = token[:i] + ch + token[i + 1:]
            if en_dict.check(cand):
                out.add(cand)

    # delete
    for i in range(n):
        cand = token[:i] + token[i + 1:]
        if len(cand) >= 3 and en_dict.check(cand):
            out.add(cand)

    # transpose
    for i in range(n - 1):
        if token[i] == token[i + 1]:
            continue
        cand = token[:i] + token[i + 1] + token[i] + token[i + 2:]
        if en_dict.check(cand):
            out.add(cand)

    return list(out)


def edit_distance_limited(a: str, b: str, max_dist: int = 2) -> int:
    # 轻量编辑距离：只用于短词纠错筛选
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        row_min = dp[0]
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # delete
                dp[j - 1] + 1,  # insert
                prev + cost,    # replace
            )
            prev = cur
            row_min = min(row_min, dp[j])
        if row_min > max_dist:
            return max_dist + 1
    return dp[lb]


def apply_dict_correction(text: str, conf: float, enable: bool) -> str:
    if not enable:
        return text
    en_dict = get_en_dict()
    if en_dict is None:
        return text

    def repl(match: re.Match) -> str:
        token = match.group(0)
        if len(token) < 4:
            return token
        low = token.lower()
        if en_dict.check(low):
            return token

        # 通用规则：若去掉末尾1个字母后是合法词，优先修正（如 Rakel -> Rake）
        if len(low) >= 5 and en_dict.check(low[:-1]):
            fixed = low[:-1]
            if token.isupper():
                return fixed.upper()
            if token[0].isupper():
                return fixed.capitalize()
            return fixed

        suggestions = en_dict.suggest(low)
        # 额外候选：单步编辑可达的合法词，覆盖中间错字（如 dyty -> duty）
        suggestions.extend(one_edit_valid_words(low, en_dict))
        # 去重并保持顺序
        seen = set()
        uniq_suggestions = []
        for s in suggestions:
            sl = str(s).lower()
            if sl in seen:
                continue
            seen.add(sl)
            uniq_suggestions.append(str(s))
        suggestions = uniq_suggestions
        # 优先选择编辑距离最小的建议，覆盖中间字符错误（如 Dyty -> Duty）
        best_by_dist = None
        best_dist = 3
        for cand in suggestions[:20]:
            c = cand.lower()
            if not c.isalpha():
                continue
            d = edit_distance_limited(low, c, max_dist=2)
            if d < best_dist:
                best_dist = d
                best_by_dist = cand
                if d == 1:
                    break
        if best_by_dist is not None and best_dist <= 1:
            fixed = best_by_dist
            if token.isupper():
                return fixed.upper()
            if token[0].isupper():
                return fixed.capitalize()
            return fixed

        fixed = choose_best_suggestion(low, suggestions)
        if fixed == low:
            return token
        if token.isupper():
            return fixed.upper()
        if token[0].isupper():
            return fixed.capitalize()
        return fixed

    return re.sub(r"\b[A-Za-z]{4,}\b", repl, text)


def text_quality(text: str, conf: float) -> float:
    core_len = len(re.sub(r"[^A-Za-z0-9]+", "", text))
    word_cnt = len([w for w in text.split(" ") if w])
    return conf * 2.0 + core_len * 0.03 + word_cnt * 0.02


def pick_cluster_best_text(cluster: Dict[str, object]) -> str:
    variants: Dict[str, Dict[str, float]] = cluster["variants"]
    best_text = str(cluster["best_text"])
    best_score = -1.0
    for txt, info in variants.items():
        count = float(info["count"])
        conf_avg = float(info["conf_sum"]) / max(1.0, count)
        core_len = float(info["core_len"])
        vote_score = count * 2.0 + conf_avg * 1.2 + core_len * 0.02
        if vote_score > best_score:
            best_score = vote_score
            best_text = txt
    return best_text


def dedupe_in_frame(lines: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    deduped: List[Tuple[str, float]] = []
    seen = set()
    for txt, cf in lines:
        key = re.sub(r"\s+", " ", txt).strip().lower()
        if key in seen:
            continue
        deduped.append((txt, cf))
        seen.add(key)
    return deduped


def white_consensus_lines(job: Dict[str, object], current_frame_idx: int) -> List[Tuple[str, float]]:
    history: List[Dict[str, object]] = job["white_history"]
    if not history:
        return []

    window = int(job["white_vote_window"])
    min_hits = int(job["white_vote_min_hits"])
    recent = history[-window:]

    candidates: List[Tuple[int, str, float]] = []
    for entry in recent:
        fidx = int(entry["frame"])
        for txt, cf in entry["lines"]:
            candidates.append((fidx, str(txt), float(cf)))

    if not candidates:
        return []

    clusters: List[Dict[str, object]] = []
    for fidx, txt, cf in candidates:
        hit = None
        for c in clusters:
            if mergeable(txt, str(c["rep"])):
                hit = c
                break
        if hit is None:
            clusters.append(
                {
                    "rep": txt,
                    "best_text": txt,
                    "best_score": text_quality(txt, cf),
                    "count": 1,
                    "conf_sum": cf,
                    "last_frame": fidx,
                }
            )
        else:
            hit["count"] = int(hit["count"]) + 1
            hit["conf_sum"] = float(hit["conf_sum"]) + cf
            hit["last_frame"] = max(int(hit["last_frame"]), fidx)
            score = text_quality(txt, cf)
            if score > float(hit["best_score"]):
                hit["best_score"] = score
                hit["best_text"] = txt
            old_core = len(re.sub(r"[^A-Za-z0-9]+", "", str(hit["rep"])))
            new_core = len(re.sub(r"[^A-Za-z0-9]+", "", txt))
            if new_core >= old_core:
                hit["rep"] = txt

    out: List[Tuple[str, float]] = []
    for c in clusters:
        # 只输出当前帧仍在出现且命中次数满足阈值的白字结果
        if int(c["last_frame"]) != current_frame_idx:
            continue
        if int(c["count"]) < min_hits:
            continue
        conf_avg = float(c["conf_sum"]) / max(1, int(c["count"]))
        out.append((str(c["best_text"]), conf_avg))

    return dedupe_in_frame(out)


def init_job_state() -> None:
    st.session_state.job = {
        "active": False,
        "paused": False,
        "finished": False,
        "video_path": "",
        "video_name": "",
        "use_gpu": True,
        "sample_every_n_frames": 5,
        "min_core_chars": 2,
        "conf_threshold": 0.5,
        "dedupe_global": True,
        "min_stable_hits": 2,
        "white_vote_window": 7,
        "white_vote_min_hits": 2,
        "use_dict_correction": True,
        "frame_idx": 0,
        "total_frames": 0,
        "rows": [],
        "recent_texts": [],
        "seen_texts": [],
        "clusters": [],
        "white_history": [],
        "started_at": None,
    }


def ensure_job_state() -> None:
    if "job" not in st.session_state:
        init_job_state()


def start_job(
    video_path: str,
    use_gpu: bool,
    sample_every_n_frames: int,
    min_core_chars: int,
    conf_threshold: float,
    dedupe_global: bool,
    min_stable_hits: int,
    white_vote_window: int,
    white_vote_min_hits: int,
    use_dict_correction: bool,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    st.session_state.job = {
        "active": True,
        "paused": False,
        "finished": False,
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "use_gpu": use_gpu,
        "sample_every_n_frames": sample_every_n_frames,
        "min_core_chars": min_core_chars,
        "conf_threshold": conf_threshold,
        "dedupe_global": dedupe_global,
        "min_stable_hits": min_stable_hits,
        "white_vote_window": white_vote_window,
        "white_vote_min_hits": white_vote_min_hits,
        "use_dict_correction": use_dict_correction,
        "frame_idx": 0,
        "total_frames": total_frames,
        "rows": [],
        "recent_texts": [],
        "seen_texts": [],
        "clusters": [],
        "white_history": [],
        "started_at": datetime.now().isoformat(),
    }


def finalize_old_clusters(job: Dict[str, object], current_frame_idx: int, force: bool = False) -> None:
    clusters: List[Dict[str, object]] = job["clusters"]
    rows: List[Dict[str, str]] = job["rows"]
    recent_texts = deque(job["recent_texts"], maxlen=12)
    seen_texts = set(job["seen_texts"])

    finalize_gap = max(int(job["sample_every_n_frames"]) * int(job["white_vote_window"]), 12)

    for cluster in clusters:
        if cluster["emitted"]:
            continue

        inactive = current_frame_idx - int(cluster["last_frame"])
        if not force and inactive < finalize_gap:
            continue

        if int(cluster["hits"]) < int(job["min_stable_hits"]):
            if float(cluster["best_score"]) < 1.95:
                cluster["emitted"] = True
                continue

        out_text = pick_cluster_best_text(cluster)
        if out_text in recent_texts:
            cluster["emitted"] = True
            continue
        if bool(job["dedupe_global"]) and out_text in seen_texts:
            cluster["emitted"] = True
            continue

        rows.append({"视频名称": str(job["video_name"]), "视频原字幕": out_text})
        recent_texts.append(out_text)
        seen_texts.add(out_text)
        cluster["emitted"] = True

    job["recent_texts"] = list(recent_texts)
    job["seen_texts"] = list(seen_texts)


def process_batch(batch_samples: int = 35) -> None:
    job = st.session_state.job
    if not job["active"] or job["paused"] or job["finished"]:
        return

    video_path = str(job["video_path"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    reader = get_ocr_reader(bool(job["use_gpu"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    frame_idx = int(job["frame_idx"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    sampled = 0
    while sampled < batch_samples:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % int(job["sample_every_n_frames"]) != 0:
            frame_idx += 1
            continue

        gray_lines: List[Tuple[str, float]] = []
        white_lines_raw: List[Tuple[str, float]] = []
        for roi, mode in get_target_rois(frame):
            extracted = extract_lines_from_roi(
                reader=reader,
                roi_bgr=roi,
                mode=mode,
                min_core_chars=int(job["min_core_chars"]),
                conf_threshold=float(job["conf_threshold"]),
            )
            if mode == "white":
                white_lines_raw.extend(extracted)
            else:
                gray_lines.extend(extracted)

        gray_lines = dedupe_in_frame(gray_lines)
        white_lines_raw = dedupe_in_frame(white_lines_raw)
        if bool(job["use_dict_correction"]):
            white_lines_raw = [
                (apply_dict_correction(txt, cf, True), cf) for txt, cf in white_lines_raw
            ]
            white_lines_raw = dedupe_in_frame(white_lines_raw)

        history: List[Dict[str, object]] = job["white_history"]
        history.append({"frame": frame_idx, "lines": white_lines_raw})
        if len(history) > 120:
            del history[:-120]

        white_lines = white_consensus_lines(job, frame_idx)
        line_texts = dedupe_in_frame(gray_lines + white_lines)

        clusters: List[Dict[str, object]] = job["clusters"]
        for line_text, line_conf in line_texts:
            hit = None
            for cluster in clusters:
                if mergeable(line_text, str(cluster["rep_text"])):
                    hit = cluster
                    break

            score = text_quality(line_text, line_conf)
            if hit is None:
                clusters.append(
                    {
                        "rep_text": line_text,
                        "best_text": line_text,
                        "best_score": score,
                        "hits": 1,
                        "last_frame": frame_idx,
                        "emitted": False,
                        "variants": {
                            line_text: {
                                "count": 1.0,
                                "conf_sum": float(line_conf),
                                "core_len": float(len(re.sub(r"[^A-Za-z0-9]+", "", line_text))),
                            }
                        },
                    }
                )
            else:
                hit["hits"] = int(hit["hits"]) + 1
                hit["last_frame"] = frame_idx
                if score > float(hit["best_score"]):
                    hit["best_score"] = score
                    hit["best_text"] = line_text

                old_core = len(re.sub(r"[^A-Za-z0-9]+", "", str(hit["rep_text"])))
                new_core = len(re.sub(r"[^A-Za-z0-9]+", "", line_text))
                if new_core >= old_core:
                    hit["rep_text"] = line_text

                variants: Dict[str, Dict[str, float]] = hit["variants"]
                if line_text not in variants:
                    variants[line_text] = {
                        "count": 0.0,
                        "conf_sum": 0.0,
                        "core_len": float(len(re.sub(r"[^A-Za-z0-9]+", "", line_text))),
                    }
                variants[line_text]["count"] += 1.0
                variants[line_text]["conf_sum"] += float(line_conf)

        finalize_old_clusters(job, frame_idx, force=False)

        sampled += 1
        frame_idx += 1

    cap.release()

    job["frame_idx"] = frame_idx
    if frame_idx >= int(job["total_frames"]):
        finalize_old_clusters(job, frame_idx, force=True)
        job["finished"] = True
        job["active"] = False


def render_results() -> None:
    job = st.session_state.job
    df = pd.DataFrame(job["rows"], columns=["视频名称", "视频原字幕"])
    st.dataframe(df, use_container_width=True, height=420)

    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="导出 CSV",
            data=csv_data,
            file_name="subtitle_results.csv",
            mime="text/csv",
        )

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="results")
        st.download_button(
            label="导出 Excel",
            data=bio.getvalue(),
            file_name="subtitle_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def reset_job() -> None:
    init_job_state()


ensure_job_state()
job = st.session_state.job

with st.sidebar:
    st.subheader("参数配置")
    video_path_input = st.text_input(
        "本地视频绝对路径",
        value=r"D:\视频\your_video.mp4",
        help="示例: D:\\videos\\demo.mp4",
    )
    use_gpu = st.checkbox("启用 EasyOCR GPU", value=True)
    sample_every_n_frames = st.number_input("每 N 帧识别一次", min_value=1, max_value=60, value=5, step=1)
    min_core_chars = st.number_input("最少有效字符数", min_value=2, max_value=20, value=10, step=1)
    conf_threshold = st.slider("OCR置信度阈值", min_value=0.10, max_value=0.95, value=0.90, step=0.05)
    dedupe_global = st.checkbox("输出前全局去重", value=True)
    min_stable_hits = st.number_input("稳定输出最少命中次数", min_value=1, max_value=10, value=2, step=1)
    white_vote_window = st.number_input("白字投票帧数", min_value=5, max_value=10, value=8, step=1)
    white_vote_min_hits = st.number_input("白字最少命中", min_value=2, max_value=10, value=2, step=1)
    use_dict_correction = st.checkbox("启用英文字典纠错(pyEnchant)", value=True)

st.info("当前为纯字幕提取模式：灰底与白色大字分开识别，支持暂停/继续/结束。")
if use_dict_correction and get_en_dict() is None:
    st.warning("未检测到 pyenchant/en_US 词典，字典纠错将自动跳过。")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.6])
start_clicked = col1.button("开始处理", type="primary", disabled=bool(job["active"]) and not bool(job["paused"]))
pause_clicked = col2.button("暂停", disabled=not (bool(job["active"]) and not bool(job["paused"])))
resume_clicked = col3.button("继续处理", disabled=not bool(job["paused"]))
end_clicked = col4.button("结束处理（重置）")

if end_clicked:
    reset_job()
    st.rerun()

if start_clicked:
    path = ensure_video_path(video_path_input)
    if not path or not os.path.exists(path):
        st.error("请先填写有效的视频路径。")
    else:
        try:
            start_job(
                video_path=path,
                use_gpu=bool(use_gpu),
                sample_every_n_frames=int(sample_every_n_frames),
                min_core_chars=int(min_core_chars),
                conf_threshold=float(conf_threshold),
                dedupe_global=bool(dedupe_global),
                min_stable_hits=int(min_stable_hits),
                white_vote_window=int(white_vote_window),
                white_vote_min_hits=int(white_vote_min_hits),
                use_dict_correction=bool(use_dict_correction),
            )
            st.rerun()
        except Exception as exc:
            st.error(f"启动失败: {exc}")

if pause_clicked:
    st.session_state.job["paused"] = True
    st.session_state.job["active"] = False
    st.rerun()

if resume_clicked:
    st.session_state.job["paused"] = False
    st.session_state.job["active"] = True
    st.rerun()

job = st.session_state.job
if bool(job["active"]) and not bool(job["paused"]) and not bool(job["finished"]):
    try:
        process_batch(batch_samples=35)
        job = st.session_state.job
        total = max(1, int(job["total_frames"]))
        cur = min(int(job["frame_idx"]), total)
        st.progress(cur / total, text=f"处理中: {cur}/{total} 帧")
        render_results()

        if bool(job["active"]) and not bool(job["finished"]):
            st.rerun()
    except Exception as exc:
        st.error(f"处理失败: {exc}")
elif bool(job["paused"]):
    total = max(1, int(job["total_frames"]))
    cur = min(int(job["frame_idx"]), total)
    st.warning(f"已暂停：{cur}/{total} 帧")
    st.progress(cur / total)
    render_results()
elif bool(job["finished"]):
    started = job["started_at"]
    elapsed = "-"
    if started:
        elapsed = str((datetime.now() - datetime.fromisoformat(str(started))).seconds)
    st.success(f"处理完成，共提取 {len(job['rows'])} 条字幕，耗时 {elapsed} 秒。")
    st.progress(1.0)
    render_results()
else:
    st.caption("等待开始处理。")
