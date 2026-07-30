import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from similar_engine import (
    PatternMatcher, WIN_RATE_BY_TF_METHOD, TF_CSV_FILENAMES, 
    _load_price_csv, DEFAULT_SIM_THRESHOLD
)

def measure_typical_pattern_span(matcher, real_price_df, method, window_size=120,
                                  step=5, min_quality=0.88, min_samples=30):
    n = len(real_price_df)
    if n < window_size + 1:
        return None, 0, []

    n_windows_checked = 0
    n_no_extrema = 0
    n_flat_price = 0
    best_scores_seen = []
    spans = []

    for end in range(window_size, n, step):
        window_df = real_price_df.iloc[end - window_size: end]
        n_windows_checked += 1

        if method == 'ZigZag Limit':
            extrema_5 = matcher.extract_extrema_zigzag(window_df, limit=0.005)
        elif method == 'Pivot Window':
            extrema_5 = matcher.extract_extrema_window(window_df, window=4)
        else:
            extrema_5 = matcher.extract_extrema_ema_peaks(window_df, span=5)

        if not extrema_5:
            n_no_extrema += 1
            continue

        prices = np.array([x[1] for x in extrema_5])
        p_min, p_max = prices.min(), prices.max()
        if p_max == p_min:
            n_flat_price += 1
            continue
        normalized = (prices - p_min) / (p_max - p_min)

        best_score = 0.0
        for template in matcher.templates.values():
            dist = euclidean(normalized, template)
            score = 1 - dist / matcher._MAX_EUCLIDEAN_5D
            best_score = max(best_score, score)
        best_scores_seen.append(best_score)

        if best_score >= min_quality:
            span = extrema_5[-1][0] - extrema_5[0][0]
            if span > 0:
                spans.append(span)

    if len(spans) < min_samples:
        return None, len(spans), spans

    return int(round(np.median(spans))), len(spans), spans

def calibrate_thresholds_for_matcher(matcher, real_price_df, method,
                                      block_size=50, n_boot=5000, alpha=0.01,
                                      random_state=None, flat_threshold=DEFAULT_SIM_THRESHOLD):
    if flat_threshold is not None:
        print(f"  [i] flat_threshold={flat_threshold} -> bỏ qua bootstrap, "
              f"dùng ngưỡng cố định {flat_threshold} cho toàn bộ "
              f"{len(matcher.templates)} pattern.")
        return {name: flat_threshold for name in matcher.templates}

    if random_state is not None:
        np.random.seed(random_state)

    close = real_price_df['Close'].reset_index(drop=True)
    n = len(close)
    if n <= block_size + 1:
        print(f"  [!] Dữ liệu quá ngắn ({n} nến) so với block_size={block_size} -> bỏ qua.")
        return {name: None for name in matcher.templates}

    null_scores = {name: [] for name in matcher.templates}

    for _ in range(n_boot):
        idx = np.random.randint(0, n - block_size)
        window_close = close.iloc[idx: idx + block_size].reset_index(drop=True)
        window_df = pd.DataFrame({'Close': window_close})

        if method == 'ZigZag Limit':
            extrema = matcher.extract_extrema_zigzag(window_df, limit=0.005)
        elif method == 'Pivot Window':
            window_df['High'] = window_close
            window_df['Low'] = window_close
            extrema = matcher.extract_extrema_window(window_df, window=4)
        else:
            extrema = matcher.extract_extrema_ema_peaks(window_df, span=5)

        if not extrema:
            continue

        prices = np.array([x[1] for x in extrema])
        p_min, p_max = prices.min(), prices.max()
        if p_max == p_min:
            continue
        norm = (prices - p_min) / (p_max - p_min)

        for name, template in matcher.templates.items():
            dist = euclidean(norm, template)
            score = 1 - dist / matcher._MAX_EUCLIDEAN_5D
            null_scores[name].append(score)

    n_templates = len(matcher.templates)
    corrected_alpha = alpha / n_templates if n_templates > 0 else alpha

    thresholds = {}
    for name, scores in null_scores.items():
        if len(scores) < 200:
            print(f"  [!] '{name}': chỉ có {len(scores)} mẫu null hợp lệ (<200) -> threshold=None, cần tăng n_boot.")
            thresholds[name] = None
            continue
        thresholds[name] = float(np.percentile(scores, 100 * (1 - corrected_alpha)))

    return thresholds

def _round_up_to_multiple(x, base=5):
    import math
    if x <= 0:
        return base
    return int(base * math.ceil(x / base))

def run_full_calibration(data_folder='.', block_size='auto', n_boot=5000, alpha=0.01,
                          random_state=42, span_window_size=120, span_step=5,
                          span_min_quality=0.88, span_fallback=50,
                          flat_threshold=DEFAULT_SIM_THRESHOLD):
    all_pairs = sorted(WIN_RATE_BY_TF_METHOD.keys())
    print(f"Sẽ calibrate {len(all_pairs)} cặp (tf, method): {all_pairs}")
    if flat_threshold is not None:
        print(f"[i] flat_threshold={flat_threshold} -> mọi pattern sẽ nhận "
              f"sim_threshold cố định {flat_threshold}, bỏ qua bootstrap thống kê.")
    print()

    results = {}
    for tf, method in all_pairs:
        filepath = os.path.join(data_folder, TF_CSV_FILENAMES.get(tf, f'data_btc_price_{tf}.csv'))
        if not os.path.exists(filepath):
            print(f"[-] Bỏ qua ({tf}, {method}): không tìm thấy {filepath}")
            continue

        df = _load_price_csv(filepath)
        if df is None:
            print(f"[-] Bỏ qua ({tf}, {method}): dữ liệu không hợp lệ")
            continue

        if block_size == 'auto':
            entry_patterns = list(WIN_RATE_BY_TF_METHOD[(tf, method)].keys())
            span_matcher = PatternMatcher(allowed_patterns=entry_patterns)
            measured_span, n_span, _ = measure_typical_pattern_span(
                span_matcher, df, method, window_size=span_window_size,
                step=span_step, min_quality=span_min_quality
            )
            if measured_span is not None:
                tf_block_size = _round_up_to_multiple(measured_span, 5)
                print(f"  [i] ({tf}, {method}): đo được {n_span} lần pattern xuất hiện "
                      f"thật -> block_size = {measured_span} -> {tf_block_size} nến.")
            else:
                tf_block_size = _round_up_to_multiple(span_fallback, 5)
                print(f"  [!] ({tf}, {method}): chưa đủ mẫu -> dùng span_fallback={span_fallback} nến.")
        else:
            tf_block_size = _round_up_to_multiple(block_size, 5)

        if len(df) <= tf_block_size + 1:
            print(f"[-] Bỏ qua ({tf}, {method}): dữ liệu không hợp lệ hoặc quá ngắn")
            continue

        entry = WIN_RATE_BY_TF_METHOD[(tf, method)]
        matcher = PatternMatcher(allowed_patterns=list(entry.keys()))
        print(f"[*] Đang calibrate ({tf}, {method}) — {len(matcher.templates)} pattern, "
              f"n_boot={n_boot}, block_size={tf_block_size}...")
        
        thresholds = calibrate_thresholds_for_matcher(
            matcher, df, method, block_size=tf_block_size, n_boot=n_boot,
            alpha=alpha, random_state=random_state, flat_threshold=flat_threshold
        )
        results[(tf, method)] = thresholds

    print("\n" + "=" * 70)
    print("KẾT QUẢ — copy phần 'sim_threshold' bên dưới vào đúng chỗ trong")
    print("WIN_RATE_BY_TF_METHOD (bên trong file similar_engine.py):")
    print("=" * 70)
    for (tf, method), thresholds in results.items():
        print(f"\n    ('{tf}', '{method}'): {{")
        for name, th in thresholds.items():
            th_str = f"{th:.4f}" if th is not None else "None"
            print(f"        '{name}': {{'sim_threshold': {th_str}}},")
        print("    },")

    return results

if __name__ == "__main__":
    import sys
    if '--calibrate' in sys.argv:
        args = [a for a in sys.argv[1:] if a != '--calibrate']
        folder = args[0] if len(args) >= 1 else '.'
        n_boot = int(args[1]) if len(args) >= 2 else 5000
        run_full_calibration(data_folder=folder, n_boot=n_boot)
    else:
        print(__doc__)
        print("Dùng lệnh: python calibrate_engine.py --calibrate [thư_mục_data] [n_boot]")