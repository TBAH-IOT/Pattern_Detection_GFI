import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(CURRENT_DIR) == 'Code':
    BASE_DIR = os.path.dirname(CURRENT_DIR) 
    DATA_DIR = os.path.join(BASE_DIR, 'Data')
else:
    DATA_DIR = os.path.join(CURRENT_DIR, 'Data')

TF_CSV_FILENAMES = {
    '15m': os.path.join(DATA_DIR, 'data_btc_price_15m.csv'),
    '1h': os.path.join(DATA_DIR, 'data_btc_price_1h.csv'), 
    '4h': os.path.join(DATA_DIR, 'data_btc_price_4h.csv'),
}
# -------------------------------------------------------------

# Các hằng số Fibonacci/retracement chuẩn
FIB_236 = 0.236
FIB_382 = 0.382
FIB_500 = 0.500
FIB_618 = 0.618
FIB_786 = 0.786


WIN_RATE_BY_TF_METHOD = {
    ('1h', 'EMA Peaks'): {
        'Diamond_Top': {'win_rate': 0.5590, 'direction_override': 'Up', 'sim_threshold': None},
        'Triangle_Ascending': {'win_rate': 0.6129, 'direction_override': None, 'sim_threshold': None},
    },
    ('1h', 'Pivot Window'): {
        'Diamond_Top': {'win_rate': 0.5763, 'direction_override': 'Up', 'sim_threshold': None},
        'Broadening_Bottom': {'win_rate': 0.5554, 'direction_override': 'Momentum', 'sim_threshold': None},
    },
    ('1h', 'ZigZag Limit'): {
        'Triple_Top_Wide': {'win_rate': 0.6037, 'direction_override': None, 'sim_threshold': None},
    },
    ('15m', 'ZigZag Limit'): {
        'Wedge_Falling': {'win_rate': 0.5688, 'direction_override': None, 'sim_threshold': None},
        'Triple_Bottom_Wide': {'win_rate': 0.6234, 'direction_override': None, 'sim_threshold': None},
        'Cup_and_Handle': {'win_rate': 0.5698, 'direction_override': 'Down', 'sim_threshold': None},
        'Double_Bottom': {'win_rate': 0.5603, 'direction_override': 'Down', 'sim_threshold': None},
        'Triple_Top': {'win_rate': 0.5849, 'direction_override': None, 'sim_threshold': None},
        'Head_and_Shoulders_Top': {'win_rate': 0.5575, 'direction_override': None, 'sim_threshold': None},
        'Head_and_Shoulders_Bottom': {'win_rate': 0.5781, 'direction_override': None, 'sim_threshold': None},
    },
    ('15m', 'Pivot Window'): {
        'Pennant_Bearish': {'win_rate': 0.5546, 'direction_override': 'Up', 'sim_threshold': None},
    },
    ('4h', 'ZigZag Limit'): {
        'Broadening_Bottom': {'win_rate': 0.5638, 'direction_override': 'Momentum', 'sim_threshold': None},
    },
    ('4h', 'Pivot Window'): {
        'Wedge_Rising': {'win_rate': 0.6607, 'direction_override': None, 'sim_threshold': None},
    },
}

DEFAULT_SIM_THRESHOLD = 0.85
_TF_WITH_SIGNAL = sorted(set(tf for tf, _ in WIN_RATE_BY_TF_METHOD.keys()))

class PatternMatcher:
    def __init__(self, allowed_patterns=None):
        self.templates = {
            'Head_and_Shoulders_Top': np.array([FIB_618, 0.0, 1.0, 0.0, FIB_618]),
            'Head_and_Shoulders_Bottom': np.array([1 - FIB_618, 1.0, 0.0, 1.0, 1 - FIB_618]),
            'Double_Top': np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
            'Double_Bottom': np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
            'Triple_Top': np.array([1 - 0.05, 0.0, 1.0, 0.05, 1 - 0.05]),
            'Triple_Bottom': np.array([0.05, 1.0, 0.0, 1 - 0.05, 0.05]),
            'Triangle_Symmetrical': np.array([1.0, 0.0, FIB_618, FIB_382, FIB_500]),
            'Triangle_Ascending': np.array([1.0, FIB_236, 1 - 0.05, FIB_618, 1.0]),
            'Triangle_Descending': np.array([0.0, 1 - FIB_236, 0.05, 1 - FIB_618, 0.0]),
            'Wedge_Rising': np.array([FIB_382, 0.0, FIB_618, FIB_500, FIB_786]),
            'Wedge_Falling': np.array([1 - FIB_382, 1.0, 1 - FIB_618, 1 - FIB_500, 1 - FIB_786]),
            'Flag_Bullish': np.array([0.0, 1.0, 1 - FIB_500, 1 - FIB_382, 0.60]),
            'Flag_Bearish': np.array([1.0, 0.0, FIB_500, FIB_382, 0.40]),
            'Pennant_Bullish': np.array([0.0, 1.0, FIB_618, FIB_786, 0.70]),
            'Pennant_Bearish': np.array([1.0, 0.0, 1 - FIB_618, 1 - FIB_786, 0.30]),
            'Rectangle_Range': np.array([0.0, 1.0, 0.05, 1 - 0.05, 0.0]),
            'Rounding_Bottom': np.array([1.0, 1 - np.sin(np.pi / 4), 0.0, 1 - np.sin(np.pi / 4), 1.0]),
            'Rounding_Top': np.array([0.0, np.sin(np.pi / 4), 1.0, np.sin(np.pi / 4), 0.0]),
            'Cup_and_Handle': np.array([0.9, 0.0, 1.0, 1 - FIB_382, 1.0]),
            'Diamond_Top': np.array([FIB_618, FIB_382, 1.0, 0.0, FIB_500]),
            'Diamond_Bottom': np.array([FIB_382, FIB_618, 0.0, 1.0, FIB_500]),
            'Broadening_Top': np.array([FIB_500, FIB_382, 1 - FIB_382, 0.0, 1.0]),
            'Broadening_Bottom': np.array([1 - FIB_500, 1 - FIB_382, FIB_382, 1.0, 0.0]),
            'Triple_Bottom_Wide': np.array([0.10, 1.0, 0.0, 1 - 0.10, 0.10]),
            'Triple_Top_Wide': np.array([1 - 0.10, 0.0, 1.0, 0.10, 1 - 0.10]),
        }

        self.stats = {
            'Head_and_Shoulders_Top': {'Direction': 'Down', 'Target': 'Biên độ = Đầu -> Cổ'},
            'Head_and_Shoulders_Bottom': {'Direction': 'Up', 'Target': 'Biên độ = Đầu -> Cổ'},
            'Double_Top': {'Direction': 'Down', 'Target': 'Biên độ = Đỉnh cao nhất'},
            'Double_Bottom': {'Direction': 'Up', 'Target': 'Biên độ = Đáy thấp nhất'},
            'Triple_Top': {'Direction': 'Down', 'Target': 'Biên độ = Đỉnh -> Đáy giữa'},
            'Triple_Bottom': {'Direction': 'Up', 'Target': 'Biên độ = Đáy -> Đỉnh giữa'},
            'Triangle_Symmetrical': {'Direction': 'Breakout', 'Target': 'Biên độ = Cạnh đáy tam giác'},
            'Triangle_Ascending': {'Direction': 'Up', 'Target': 'Biên độ = Chiều cao tam giác'},
            'Triangle_Descending': {'Direction': 'Down', 'Target': 'Biên độ = Chiều cao tam giác'},
            'Wedge_Rising': {'Direction': 'Down', 'Target': 'Biên độ = Điểm bắt đầu nêm'},
            'Wedge_Falling': {'Direction': 'Up', 'Target': 'Biên độ = Điểm bắt đầu nêm'},
            'Flag_Bullish': {'Direction': 'Up', 'Target': 'Biên độ = Chiều dài cột cờ'},
            'Flag_Bearish': {'Direction': 'Down', 'Target': 'Biên độ = Chiều dài cột cờ'},
            'Pennant_Bullish': {'Direction': 'Up', 'Target': 'Biên độ = Cột cờ trước đó'},
            'Pennant_Bearish': {'Direction': 'Down', 'Target': 'Biên độ = Cột cờ trước đó'},
            'Rectangle_Range': {'Direction': 'Breakout', 'Target': 'Biên độ = Chiều cao hộp giá'},
            'Rounding_Bottom': {'Direction': 'Up', 'Target': 'Biên độ = Độ sâu đáy bo tròn'},
            'Rounding_Top': {'Direction': 'Down', 'Target': 'Biên độ = Độ cao đỉnh bo tròn'},
            'Cup_and_Handle': {'Direction': 'Up', 'Target': 'Biên độ = Độ sâu Cup'},
            'Diamond_Top': {'Direction': 'Down', 'Target': 'Biên độ = Chiều rộng Diamond'},
            'Diamond_Bottom': {'Direction': 'Up', 'Target': 'Biên độ = Chiều rộng Diamond'},
            'Broadening_Top': {'Direction': 'Breakout', 'Target': 'Biên độ = Sóng dao động lớn nhất'},
            'Broadening_Bottom': {'Direction': 'Breakout', 'Target': 'Biên độ = Sóng dao động lớn nhất'},
            'Triple_Bottom_Wide': {'Direction': 'Up', 'Target': 'Biên độ = Đáy -> Đỉnh giữa (dung sai rộng)'},
            'Triple_Top_Wide': {'Direction': 'Down', 'Target': 'Biên độ = Đỉnh -> Đáy giữa (dung sai rộng)'},
        }

        if allowed_patterns is not None:
            self.templates = {k: v for k, v in self.templates.items() if k in allowed_patterns}
            self.stats = {k: v for k, v in self.stats.items() if k in allowed_patterns}

        self.win_rates = {}  
        self.direction_overrides = {} 
        self.sim_thresholds = {} 

    @classmethod
    def for_tf_method(cls, tf, method):
        entry = WIN_RATE_BY_TF_METHOD.get((tf, method), {})
        matcher = cls(allowed_patterns=list(entry.keys()))
        matcher.win_rates = {name: v['win_rate'] for name, v in entry.items()}
        matcher.direction_overrides = {name: v['direction_override'] for name, v in entry.items()}
        matcher.sim_thresholds = {
            name: v['sim_threshold'] for name, v in entry.items()
            if v.get('sim_threshold') is not None
        }
        return matcher

    def resolve_direction(self, pattern_name, entry_price, last_extrema_price):
        base_direction = self.stats[pattern_name]['Direction']
        override = self.direction_overrides.get(pattern_name)

        if override == 'Momentum':
            return 'Up' if entry_price >= last_extrema_price else 'Down'
        elif override in ('Up', 'Down'):
            return override
        return base_direction

    def extract_extrema_zigzag(self, df_view, limit=0.005):
        if len(df_view) < 2: return None
        pivots = []
        pivot_state = 'down'
        ext_price = df_view['Close'].iloc[0]
        ext_idx = 0
        
        for i in range(1, len(df_view)):
            curr_close = df_view['Close'].iloc[i]
            if pivot_state == 'down':
                if curr_close <= ext_price:
                    ext_price = curr_close
                    ext_idx = i
                elif curr_close >= ext_price * (1 + limit):
                    pivots.append((ext_idx, ext_price, 'trough'))
                    pivot_state = 'up'
                    ext_price = curr_close
                    ext_idx = i
            else:
                if curr_close >= ext_price:
                    ext_price = curr_close
                    ext_idx = i
                elif curr_close <= ext_price * (1 - limit):
                    pivots.append((ext_idx, ext_price, 'peak'))
                    pivot_state = 'down'
                    ext_price = curr_close
                    ext_idx = i
                    
        if len(pivots) >= 5: return pivots[-5:]
        return None

    def extract_extrema_window(self, df_view, window=4):
        if len(df_view) < window * 2 + 1: return None
        highs = df_view['High'].values
        lows = df_view['Low'].values
        extrema = []
        
        for i in range(window, len(df_view) - window):
            if highs[i] == np.max(highs[i-window : i+window+1]):
                extrema.append((i, highs[i], 'peak'))
            if lows[i] == np.min(lows[i-window : i+window+1]):
                extrema.append((i, lows[i], 'trough'))
                
        extrema.sort(key=lambda x: x[0])
        if len(extrema) < 5: return None
        
        alt = [extrema[0]]
        for ext in extrema[1:]:
            if ext[2] != alt[-1][2]: alt.append(ext)
            else:
                if ext[2] == 'peak' and ext[1] > alt[-1][1]: alt[-1] = ext
                elif ext[2] == 'trough' and ext[1] < alt[-1][1]: alt[-1] = ext
                
        if len(alt) >= 5: return alt[-5:]
        return None

    def extract_extrema_ema_peaks(self, df_view, span=5):
        smoothed = df_view['Close'].ewm(span=span, adjust=False).mean().values
        peaks, _ = find_peaks(smoothed, distance=3)
        troughs, _ = find_peaks(-smoothed, distance=3)
        
        extrema = [(p, smoothed[p], 'peak') for p in peaks] + [(t, smoothed[t], 'trough') for t in troughs]
        extrema.sort(key=lambda x: x[0])
        if len(extrema) < 5: return None
            
        alt = [extrema[0]]
        for ext in extrema[1:]:
            if ext[2] != alt[-1][2]: alt.append(ext)
            else:
                if ext[2] == 'peak' and ext[1] > alt[-1][1]: alt[-1] = ext
                elif ext[2] == 'trough' and ext[1] < alt[-1][1]: alt[-1] = ext
                
        if len(alt) >= 5: return alt[-5:]
        return None

    _MAX_EUCLIDEAN_5D = np.sqrt(5)

    def search_pattern(self, df_view, method='EMA Peaks', default_threshold=DEFAULT_SIM_THRESHOLD, verbose_fallback=False):
        if method == 'ZigZag Limit':
            extrema_5 = self.extract_extrema_zigzag(df_view, limit=0.005)
        elif method == 'Pivot Window':
            extrema_5 = self.extract_extrema_window(df_view, window=4)
        else: 
            extrema_5 = self.extract_extrema_ema_peaks(df_view, span=5)
            
        if not extrema_5: return "None", 0.0, None
            
        prices = np.array([x[1] for x in extrema_5])
        p_min, p_max = np.min(prices), np.max(prices)
        if p_max == p_min: return "None", 0.0, None
            
        normalized_prices = (prices - p_min) / (p_max - p_min)
        
        best_match = "None"
        highest_score = 0.0
        best_score_seen = 0.0 

        for name, template in self.templates.items():
            dist = euclidean(normalized_prices, template)
            sim_score = 1 - (dist / self._MAX_EUCLIDEAN_5D)
            best_score_seen = max(best_score_seen, sim_score)

            pattern_threshold = self.sim_thresholds.get(name)
            if pattern_threshold is None:
                pattern_threshold = default_threshold
                if verbose_fallback:
                    print(f"[!] Pattern '{name}' chưa có sim_threshold calibrate riêng -> "
                          f"tạm dùng default_threshold={default_threshold:.2f}.")

            if sim_score >= pattern_threshold and sim_score > highest_score:
                highest_score = sim_score
                best_match = name

        if best_match != "None":
            return best_match, highest_score, extrema_5
        return "None", best_score_seen, None

    @staticmethod
    def match_quality_label(score):
        if score >= 0.97:
            return "Rất chuẩn"
        elif score >= 0.93:
            return "Chuẩn"
        elif score >= 0.88:
            return "Khá chuẩn"
        else:
            return "Tương đối"

# Hàm này được giữ lại vì V5 UI cần dùng nó để load file csv
def _load_price_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = [
            col.replace('\ufeff', '').strip().capitalize().replace(' ', '_')
            for col in df.columns
        ]

        TIME_COL_CANDIDATES = [
            'Timestamp', 'Date', 'Time', 'Datetime', 'Open_time', 'Opentime',
            'Date_time', 'Close_time',
        ]
        time_col = next((c for c in TIME_COL_CANDIDATES if c in df.columns), None)

        if time_col is None:
            unnamed_cols = [c for c in df.columns if c.lower().startswith('unnamed')]
            if unnamed_cols:
                candidate = unnamed_cols[0]
                parsed = pd.to_datetime(df[candidate], errors='coerce')
                if parsed.notna().mean() > 0.99:
                    df.rename(columns={candidate: 'Timestamp'}, inplace=True)
                    df['Timestamp'] = parsed
                    time_col = 'Timestamp'
                else:
                    df.drop(columns=[candidate], inplace=True)
                    df['Timestamp'] = pd.RangeIndex(len(df))
                    time_col = 'Timestamp'

        if time_col is None:
            raise KeyError(
                f"không tìm thấy cột thời gian trong {list(df.columns)} "
                f"(đã thử các tên: {TIME_COL_CANDIDATES})"
            )
        if time_col != 'Timestamp':
            df.rename(columns={time_col: 'Timestamp'}, inplace=True)

        ts_sample = df['Timestamp'].iloc[0]
        if pd.api.types.is_numeric_dtype(df['Timestamp']):
            unit = 'ms' if ts_sample > 10**12 else 's'
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit=unit)
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"  [-] Lỗi đọc {filepath}: {e}")
        return None