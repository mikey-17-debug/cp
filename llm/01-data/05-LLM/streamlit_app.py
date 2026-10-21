"""
目标：使用 streamlit 中转请求，实现模型调用
UI：马卡龙液态玻璃风格 + 动画 + 高对比度输入框
"""

import streamlit as st
import requests

# ============ 页面配置 ============
st.set_page_config(
    page_title="电影类型分类 · LLM MVP",
    page_icon="🎬",
    layout="centered",
)

API_URL = "http://127.0.0.1:8002/predict"
TIMEOUT = 30

# ============ 全局样式：马卡龙液态玻璃 + 动画 ============
st.markdown(
    """
    <style>
    /* ---------- 背景：马卡龙流动渐变 (浅粉、浅蓝、薄荷绿) ---------- */
    .stApp {
        background: linear-gradient(-45deg, #fbc2eb, #a6c1ee, #e0c3fc, #8fd3f4, #a1ffce);
        background-size: 400% 400%;
        animation: gradientFlow 20s ease infinite;
        color: #333333; /* 全局默认文字颜色改为深灰 */
    }
    @keyframes gradientFlow {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ---------- 隐藏 Streamlit 默认元素 ---------- */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    /* ---------- 标题 ---------- */
    .glass-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #2c3e50; /* 深色字体 */
        letter-spacing: 2px;
        text-shadow: 0 2px 10px rgba(255,255,255,0.8);
        margin-bottom: 0.3rem;
        animation: fadeSlideDown 1s ease both;
    }
    .glass-subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 2rem;
        animation: fadeSlideDown 1.2s ease both;
    }
    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ---------- 玻璃卡片通用样式 (浅色版) ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.35); /* 提高白色透明度 */
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.7);
        box-shadow:
            0 8px 32px rgba(100, 100, 120, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px) saturate(150%);
        -webkit-backdrop-filter: blur(20px) saturate(150%);
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
        animation: fadeUp 0.9s ease both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(25px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* 悬浮微交互 */
    .glass-card:hover {
        transform: translateY(-4px);
        transition: all 0.35s cubic-bezier(.2,.8,.2,1);
        box-shadow:
            0 14px 45px rgba(100, 100, 120, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }

    /* ---------- 解决输入框文字看不清的问题 ---------- */
    /* 标签文字 */
    .stTextArea label, .stTextArea p {
        color: #2c3e50 !important; 
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    /* 输入框本体 */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.75) !important; /* 加白背景，降低透明度 */
        color: #111111 !important; /* 输入的文字改为深黑色 */
        border-radius: 14px !important;
        border: 2px solid rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        transition: all 0.3s ease;
    }
    /* 输入框聚焦时的状态 */
    .stTextArea textarea:focus {
        border: 2px solid #a6c1ee !important; /* 聚焦时显示马卡龙蓝边框 */
        box-shadow: 0 0 15px rgba(166, 193, 238, 0.5) !important;
        background: #ffffff !important;
    }

    /* ---------- 按钮：马卡龙色玻璃 + 动画 ---------- */
    .stButton > button {
        width: 100%;
        padding: 0.85rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50; /* 深色文字 */
        background: linear-gradient(120deg, rgba(255,255,255,0.6), rgba(255,255,255,0.3));
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        backdrop-filter: blur(14px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.35s cubic-bezier(.2,.8,.2,1);
        position: relative;
        overflow: hidden;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        background: linear-gradient(120deg, rgba(255,255,255,0.9), rgba(255,255,255,0.6));
        box-shadow: 0 8px 25px rgba(166, 193, 238, 0.6);
        color: #000;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }
    /* 按钮扫光动画 */
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0; left: -80%;
        width: 60%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.8), transparent);
        transform: skewX(-20deg);
        animation: shine 3.5s infinite;
    }
    @keyframes shine {
        0%   { left: -80%; }
        60%  { left: 130%; }
        100% { left: 130%; }
    }

    /* ---------- 结果卡片 ---------- */
    .result-card {
        background: rgba(255,255,255,0.45);
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.8);
        backdrop-filter: blur(20px) saturate(160%);
        padding: 1.6rem 1.8rem;
        margin-top: 1.4rem;
        color: #333;
        box-shadow: 0 12px 40px rgba(100, 100, 120, 0.15);
        animation: popIn 0.6s cubic-bezier(.2,.9,.3,1.4) both;
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.9) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    .result-card .label {
        font-size: 0.85rem;
        color: #666;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .result-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        /* 字色使用马卡龙深色渐变 */
        background: linear-gradient(90deg, #4facfe, #00f2fe, #ff0844);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        background-size: 200% 200%;
        margin: 0.2rem 0;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .result-card .meta {
        font-size: 0.95rem;
        color: #4a5568;
        margin-top: 0.6rem;
        font-weight: 500;
    }

    /* ---------- 三列指标 ---------- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.2rem;
    }
    .metric-cell {
        background: rgba(255,255,255,0.35);
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.3s ease;
    }
    .metric-cell:hover { transform: translateY(-4px); background: rgba(255,255,255,0.5); }
    .metric-cell .k {
        color: #555;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .metric-cell .v {
        color: #2c3e50;
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* ---------- Spinner 颜色 ---------- */
    .stSpinner > div { border-top-color: #a6c1ee !important; }
    
    /* 展开面板及警告框适配浅色背景 */
    .stAlert { background-color: rgba(255,255,255,0.7); border: 1px solid #fff; color: #333; }
    .stExpander { background-color: rgba(255,255,255,0.4); border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ 头部 ============
st.markdown(
    """
    <div class="glass-title">🎬 电影类型分类 · LLM MVP</div>
    <div class="glass-subtitle">输入电影片名和剧情简介，让大模型帮你判断它属于哪一类</div>
    """,
    unsafe_allow_html=True,
)

# ============ 输入卡片 ============
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    title = st.text_area(
        label="📝 请输入电影片名和剧情简介",
        value="你的名字。巫女世家出身的高中女孩宫水三叶在梦中变成东京的高中男孩立花泷，两人交换身份后发生了一系列奇幻的故事。",
        height=150,
    )
    predict_clicked = st.button("✨ 开始预测", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# ============ 逻辑处理 ============
if predict_clicked:
    if not title.strip():
        st.warning("请输入电影片名或剧情简介后再进行预测。")
        st.stop()

    try:
        with st.spinner("正在调用模型进行预测..."):
            response = requests.post(API_URL, json={"text": title}, timeout=TIMEOUT)
        response.raise_for_status()

        try:
            result = response.json()
        except ValueError:
            st.error("后端返回的不是合法 JSON。")
            st.code(response.text)
            st.stop()

        pred_class = result.get("pred_class", "-")
        duration = result.get("duration", "-")
        text_back = result.get("text", "")

        # 结果卡片
        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Prediction</div>
                <div class="value">{pred_class}</div>
                <div class="meta">📄 预测文本：{text_back}</div>
            </div>
            <div class="metric-grid">
                <div class="metric-cell">
                    <div class="k">预测耗时</div>
                    <div class="v">{duration} s</div>
                </div>
                <div class="metric-cell">
                    <div class="k">文本长度</div>
                    <div class="v">{len(text_back)} 字</div>
                </div>
                <div class="metric-cell">
                    <div class="k">状态</div>
                    <div class="v">✅ 成功</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔍 查看原始返回"):
            st.json(result)

    except requests.exceptions.Timeout:
        st.error(f"请求超时（>{TIMEOUT}s），请检查模型服务。")
    except requests.exceptions.ConnectionError:
        st.error(f"无法连接后端服务：{API_URL}")
    except requests.exceptions.HTTPError as e:
        st.error(f"后端返回错误：{e.response.status_code}")
        st.code(e.response.text)
    except Exception as e:
        st.error(f"未知错误：{e}")
