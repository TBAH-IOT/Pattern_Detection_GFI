import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# ==========================================
# "NÃO" CỦA HỆ THỐNG -> LẤY TỪ pattern_engine.py (đúng logic của V4)
# ==========================================
# File này KHÔNG còn tự định nghĩa PatternMatcher/WIN_RATE_BY_TF_METHOD riêng
# nữa (bản cũ dùng cosine similarity + ngưỡng 0.85 cứng cho mọi pattern, và
# WIN_RATE_BY_TF_METHOD bị lệch/lỗi thời so với bản đã sửa look-ahead bias
# trong V4). Toàn bộ logic nhận diện pattern (Euclidean similarity, ngưỡng
# calibrate RIÊNG từng pattern qua block-bootstrap, resolve hướng giao dịch
# có override/momentum...) giờ lấy trực tiếp từ pattern_engine.py -- cùng một
# "não" mà similarity_pattern_V4.py (giao diện Chart Replay Trainer) đang
# dùng. File này chỉ còn nhiệm vụ HIỂN THỊ GIAO DIỆN (Streamlit + Plotly).
from similar_engine import (
    PatternMatcher, WIN_RATE_BY_TF_METHOD, TF_CSV_FILENAMES, _load_price_csv,
)

# ==========================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Pro Quant - Pattern Scanner", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    
    .metric-box {
        background-color: #1E2129; padding: 20px; border-radius: 10px;
        border-left: 5px solid #26A69A; margin-bottom: 15px;
        color: #E6EDF3;
        font-size: 16px;
    }
    .metric-title { color: #8B949E; font-size: 14px; font-weight: bold; text-transform: uppercase; margin-bottom: 8px;}
    .metric-value { color: #FFFFFF; font-size: 26px; font-weight: bold; margin: 5px 0; }
    .metric-sub { color: #26A69A; font-size: 15px; font-weight: bold;}
    .metric-sub.down { color: #EF5350; border-left-color: #EF5350; }
    .metric-warn { color: #FFB74D; font-size: 13px; margin-top: 6px; }
    
    div.stButton > button {
        width: 100%;
        background-color: #2D333B;
        color: #E6EDF3;
        border: 1px solid #444C56;
    }
    div.stButton > button:hover {
        border-color: #26A69A;
        color: #26A69A;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA & SESSION STATE (DÙNG LOCAL CSV)
# ==========================================
@st.cache_data(ttl=3600)
def load_data(tf):
    """Đọc CSV giá thật cho khung `tf`. Việc parse CSV (chuẩn hoá cột,
    parse Timestamp, set index, sort) dùng đúng _load_price_csv() của
    pattern_engine -- cùng 1 hàm mà run_full_calibration() trong V4 dùng để
    đọc dữ liệu calibrate, tránh 2 cách đọc file lệch nhau."""
    filename = TF_CSV_FILENAMES.get(tf)
    if not filename:
        return pd.DataFrame()

    filepath = os.path.join('.', filename)
    df = _load_price_csv(filepath)
    if df is None:
        return pd.DataFrame()

    df.dropna(inplace=True)
    return df

st.title("⚡ Quantitative Pattern Scanner")

# Thống kê nhanh để minh bạch với người dùng: có bao nhiêu pattern đã xác
# nhận qua backtest, và bao nhiêu trong số đó CHƯA calibrate ngưỡng
# similarity riêng (đang dùng DEFAULT_SIM_THRESHOLD dự phòng).
_unique_patterns = set()
_total_slots = 0
_calibrated_slots = 0
for _entry in WIN_RATE_BY_TF_METHOD.values():
    _unique_patterns.update(_entry.keys())
    for _v in _entry.values():
        _total_slots += 1
        if _v.get('sim_threshold') is not None:
            _calibrated_slots += 1



window_size = 120

# ==========================================
# THANH CÔNG CỤ DỌN GỌN BÊN TRÊN
# ==========================================
col_tf, col_yr, col_sp = st.columns([2, 2, 6])

# Chỉ cho chọn các khung thời gian THỰC SỰ có ít nhất 1 pattern đã backtest
# trong WIN_RATE_BY_TF_METHOD (lấy từ pattern_engine) -- tránh cho người
# dùng chọn 1 khung mà hệ thống chưa có kết quả backtest nào đứng sau nó.
available_tfs_order = ['1m', '15m', '1h', '4h', '1d']
tf_with_signal = set(tf for tf, _ in WIN_RATE_BY_TF_METHOD.keys())
tf_options = [tf for tf in available_tfs_order if tf in tf_with_signal]

with col_tf:
    tf = st.selectbox("Khung thời gian", tf_options)

df = load_data(tf)
total_len = len(df)

if df.empty:
    st.error(f"Không tải được dữ liệu cho khung {tf}. Kiểm tra file "
             f"'{TF_CSV_FILENAMES.get(tf, f'data_btc_price_{tf}.csv')}' có tồn tại trong thư mục chạy không.")
    st.stop()

# Lấy danh sách các năm thực tế có trong dữ liệu
available_years = sorted(list(set(df.index.year)))
default_year = 2022 if 2022 in available_years else available_years[-1]

def get_year_index(df_data, target_year):
    target_date = pd.to_datetime(f'{target_year}-01-01')
    if df_data.index.tz is not None:
        target_date = target_date.tz_localize(df_data.index.tz)
    idx_pos = df_data.index.searchsorted(target_date)
    
    if window_size <= idx_pos < len(df_data):
        return int(idx_pos)
    return min(window_size, len(df_data) - 1)

# Reset hoặc khởi tạo state khi đổi TF
if 'current_bar' not in st.session_state or st.session_state.get('last_tf') != tf:
    st.session_state.current_bar = get_year_index(df, default_year)
    st.session_state.last_tf = tf
    st.session_state.selected_year_key = default_year

# Callback khi đổi năm trên dropdown
def on_year_change():
    chosen_year = st.session_state.selected_year_key
    st.session_state.current_bar = get_year_index(df, chosen_year)

with col_yr:
    st.selectbox(
        "Năm Replay", 
        options=available_years, 
        key="selected_year_key", 
        on_change=on_year_change
    )

# Các hàm điều khiển Replay
def step_backward():
    st.session_state.current_bar = max(window_size, st.session_state.current_bar - 1)
    new_year = df.index[st.session_state.current_bar].year
    if new_year in available_years:
        st.session_state.selected_year_key = new_year

def step_forward():
    st.session_state.current_bar = min(total_len - 1, st.session_state.current_bar + 1)
    new_year = df.index[st.session_state.current_bar].year
    if new_year in available_years:
        st.session_state.selected_year_key = new_year

def go_live():
    st.session_state.current_bar = total_len - 1
    new_year = df.index[st.session_state.current_bar].year
    if new_year in available_years:
        st.session_state.selected_year_key = new_year

# Cắt DataFrame
current_bar = st.session_state.current_bar
start_idx = max(0, current_bar - window_size + 1)
df_view = df.iloc[start_idx : current_bar + 1].copy()

# ==========================================
# GIAO DIỆN NÚT BẤM ĐIỀU KHIỂN NẾN
# ==========================================
st.markdown("---")
b1, b2, b3, b_empty = st.columns([1.2, 1.2, 1.5, 6.1])
with b1:
    st.button("⏪ Lùi 1 nến", on_click=step_backward, use_container_width=True)
with b2:
    st.button("Tiến 1 nến ⏩", on_click=step_forward, use_container_width=True)
with b3:
    st.button("⏭️ Tới Live", on_click=go_live, use_container_width=True)

current_date_str = df_view.index[-1].strftime('%Y-%m-%d %H:%M')
is_live = "ĐANG CHẠY LIVE" if current_bar == total_len - 1 else f"REPLAY QUÁ KHỨ ({current_date_str})"
st.caption(f"Trạng thái: **{is_live}** | Nến hiện tại: **{current_date_str}**")

# ==========================================
# THUẬT TOÁN QUÉT TÍN HIỆU -> GỌI THẲNG "NÃO" (PatternMatcher.search_pattern)
# ==========================================
# search_pattern() ở đây là bản Euclidean similarity + ngưỡng calibrate
# RIÊNG từng pattern (self.sim_thresholds, tự tra bên trong hàm) của
# pattern_engine -- KHÔNG còn threshold=0.85 cứng như bản V5 cũ.
methods = ['EMA Peaks', 'Pivot Window', 'ZigZag Limit']
best_signal = None
best_score_overall = 0

for method in methods:
    matcher = PatternMatcher.for_tf_method(tf, method)
    if not matcher.templates:
        continue

    p_name, score, ext = matcher.search_pattern(df_view, method=method)
    if p_name != "None" and score > best_score_overall:
        last_ext_idx = ext[-1][0]
        current_idx = len(df_view) - 1

        # Chỉ chấp nhận tín hiệu còn "tươi" (cực trị cuối cùng cách nến
        # hiện tại không quá 10 nến) -- lọc riêng của giao diện, không phải
        # logic thuộc "não".
        if current_idx - last_ext_idx <= 10:
            best_score_overall = score
            best_signal = {
                'name': p_name, 'score': score, 'method': method,
                'extrema': ext, 'matcher': matcher,
            }

# ==========================================
# RENDER UI: CHART (Trái) & PANEL (Phải)
# ==========================================
col_chart, col_info = st.columns([7, 3])

with col_chart:
    fig = go.Figure(data=[go.Candlestick(
        x=df_view.index, open=df_view['Open'], high=df_view['High'],
        low=df_view['Low'], close=df_view['Close'],
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    )])
    
    if best_signal:
        ext = best_signal['extrema']
        x_vals = [df_view.index[e[0]] for e in ext]
        y_vals = [e[1] for e in ext]
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers',
            line=dict(color='#00E5FF', width=3),
            marker=dict(size=8, color='#00E5FF'),
            name='Pattern Path'
        ))
        
        entry_price = df_view['Close'].iloc[-1]
        height = max(y_vals) - min(y_vals)
        direction = best_signal['matcher'].resolve_direction(best_signal['name'], entry_price, y_vals[-1])
        
        if direction == 'Up':
            tp, sl = entry_price + height, entry_price - height
            color_tp, color_sl = '#00E676', '#FF1744'
        else:
            tp, sl = entry_price - height, entry_price + height
            color_tp, color_sl = '#FF1744', '#00E676'
            
        lookback_len = min(30, len(df_view) - 1)
        last_dates = [df_view.index[-lookback_len], df_view.index[-1]]
        
        fig.add_trace(go.Scatter(x=last_dates, y=[entry_price]*2, mode='lines', line=dict(color='white', width=1, dash='dot'), name='Entry'))
        fig.add_trace(go.Scatter(x=last_dates, y=[tp]*2, mode='lines', line=dict(color=color_tp, width=2, dash='dash'), name='TP'))
        fig.add_trace(go.Scatter(x=last_dates, y=[sl]*2, mode='lines', line=dict(color=color_sl, width=2, dash='dash'), name='SL'))

    fig.update_layout(
        template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False, height=600,
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width='stretch')

with col_info:
    st.markdown("### 📡 Tín hiệu Hệ thống")
    if best_signal:
        direction = best_signal['matcher'].resolve_direction(best_signal['name'], entry_price, best_signal['extrema'][-1][1])
        win_rate = best_signal['matcher'].win_rates.get(best_signal['name'], 0) * 100
        css_class = "metric-sub" if direction == 'Up' else "metric-sub down"
        
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: {'#00E676' if direction=='Up' else '#FF1744'};">
            <div class="metric-title">Mô hình phát hiện</div>
            <div class="metric-value">{best_signal['name']}</div>
            <div class="{css_class}">Xu hướng: {direction.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        # Ngưỡng similarity đã dùng để chấp nhận tín hiệu này: nếu pattern
        # CHƯA có sim_threshold calibrate riêng (self.sim_thresholds không
        # có key này), search_pattern() đã tự rơi về DEFAULT_SIM_THRESHOLD
        # -- báo rõ điều này ra UI thay vì âm thầm coi là đã kiểm định.
        is_calibrated = best_signal['name'] in best_signal['matcher'].sim_thresholds
        threshold_note = (
            "Ngưỡng đã calibrate riêng (block-bootstrap)" if is_calibrated
            else ""
        )
        quality_label = PatternMatcher.match_quality_label(best_signal['score'])

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Thông số Kỹ thuật</div>
            <div>Phương pháp Pivot: <b>{best_signal['method']}</b></div>
            <div>Độ khớp (Similarity): <b>{best_signal['score']*100:.2f}%</b> ({quality_label})</div>
            <div>Win Rate Lịch sử: <b>{win_rate:.1f}% ({tf})</b></div>
            <div class="metric-warn">{threshold_note}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Kế hoạch Giao dịch (R:R = 1:1)</div>
            <div>Entry: <b>{entry_price:,.2f}</b></div>
            <div style="color: #00E676; margin-top: 5px;">Take Profit (TP): <b>{tp:,.2f}</b></div>
            <div style="color: #FF1744; margin-top: 5px;">Stop Loss (SL): <b>{sl:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Trạng thái: Chưa phát hiện pattern nào (đã backtest, win rate ≥ 55%) trong cửa sổ quan sát hiện tại.")