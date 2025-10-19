import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Tích phân gần đúng", layout="wide")

st.title("Hai phương pháp tính gần đúng tích phân")
st.markdown("### Phương pháp Hình thang và Simpson")

# HÀM CHUẨN HÓA BIỂU THỨC NGƯỜI DÙNG
def normalize_expr(expr_str):
    expr_str = expr_str.lower()
    replacements = {'^': '**', 'ln': 'log', '√': 'sqrt', 'e': 'E'}
    for k, v in replacements.items():
        expr_str = expr_str.replace(k, v)
    return expr_str


# HAI CÔNG THỨC TÍNH TÍCH PHÂN
def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

def simpson_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return (h/3)*(y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))

# GIAO DIỆN NHẬP LIỆU

col1, col2 = st.columns(2)
with col1:
    expr_str = st.text_input("Nhập hàm f(x):", "x**2")
    expr_str = normalize_expr(expr_str)
    a = st.number_input("Cận dưới a:", value=0.0)
    b = st.number_input("Cận trên b:", value=1.0)

with col2:
    method = st.radio("Chọn phương pháp:", ["Hình thang", "Simpson", "Cả hai"])
    mode = st.radio("Chọn cách nhập:", ["Nhập số khoảng n", "Nhập sai số ε"])
    if mode == "Nhập số khoảng n":
        n = st.number_input("Số khoảng chia n:", min_value=2, value=10, step=1)
        epsilon = None
    else:
        epsilon = st.number_input("Sai số ε:", min_value=1e-8, value=1e-4, format="%.1e")
        n = None

# XỬ LÝ HÀM NGƯỜI DÙNG
x = sp.Symbol('x')

# Bước 1: Kiểm tra cú pháp biểu thức 
try:
    f_expr = sp.sympify(expr_str)
    f_lambda = sp.lambdify(x, f_expr, "numpy")
except Exception as e:
    st.error("Cú pháp hàm không hợp lệ. Hãy kiểm tra lại (ví dụ: sin(x), e**x, x**2, log(x), ...)")
    st.stop()

# Bước 2: Kiểm tra miền xác định trên đoạn [a, b] 
X_test = np.linspace(a, b, 400)
try:
    Y_test = f_lambda(X_test)
except Exception as e:
    st.error(f"Lỗi khi tính giá trị hàm trên đoạn [{a}, {b}]. Có thể hàm không xác định tại một số điểm.\n\nChi tiết: {e}")
    st.stop()

# Bước 3: Kiểm tra giá trị phức, vô hạn, hoặc NaN 
if np.any(np.iscomplex(Y_test)):
    st.error("Hàm trả về giá trị phức trên đoạn tích phân. "
             "Vui lòng chọn khoảng không chứa điểm khiến mẫu số âm hoặc căn của số âm.")
    st.stop()

if np.any(~np.isfinite(Y_test)):
    st.error("Hàm có giá trị vô hạn hoặc không xác định (inf / nan) trong khoảng tích phân.\n"
             "Vui lòng chọn đoạn không chứa tiệm cận hoặc điểm kỳ dị.")
    st.stop()

# Bước 4: Tính tích phân chính xác (nếu có thể) 
try:
    I_exact = float(sp.integrate(f_expr, (x, a, b)))
except Exception:
    st.warning("Không thể tính chính xác tích phân biểu tượng cho hàm này. "
               "Hệ thống sẽ chỉ so sánh kết quả gần đúng.")
    I_exact = None

# HÀM TÍNH THEO SAI SỐ HOẶC SỐ KHOẢNG

def compute_with_tolerance(f, a, b, rule_func, epsilon=None, n=None):
    prev = None
    if epsilon is not None:
        n = 2
        while True:
            I1 = rule_func(f, a, b, n)
            if prev is not None and abs(I1 - prev) < epsilon:
                break
            prev = I1
            n *= 2
    else:
        I1 = rule_func(f, a, b, n)
    return I1, n

# TÍNH KẾT QUẢ

I_trap = I_simp = None
n_used_trap = n_used_simp = None
err_trap = err_simp = None

if method in ["Hình thang", "Cả hai"]:
    I_trap, n_used_trap = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, n)
    err_trap = abs(I_trap - I_exact)

if method in ["Simpson", "Cả hai"]:
    I_simp, n_used_simp = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, n)
    err_simp = abs(I_simp - I_exact)

# HIỂN THỊ KẾT QUẢ
st.subheader("Kết quả")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.6f}")

if method in ["Hình thang", "Cả hai"]:
    cols[1].metric("Hình thang", f"{I_trap:.6f}", f"Sai số: {err_trap:.6f}")
if method in ["Simpson", "Cả hai"]:
    cols[2].metric("Simpson", f"{I_simp:.6f}", f"Sai số: {err_simp:.6f}")

# BẢNG GIÁ TRỊ

st.subheader("Bảng giá trị tại các điểm chia")

def make_table(X, Y, method):
    n = len(X) - 1
    h = (X[-1] - X[0]) / n
    if method == "Hình thang":
        weights = np.ones_like(Y)
        weights[0] = weights[-1] = 0.5
    elif method == "Simpson":
        weights = np.ones_like(Y)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
    else:
        weights = np.ones_like(Y)

    wf = weights * Y
    total = np.sum(wf)

    df = pd.DataFrame({
        "xᵢ": [f"{x:.6f}" for x in X],
        "f(xᵢ)": [f"{y:.6f}" for y in Y],
        "Trọng số (wᵢ)": [f"{w:.2f}" for w in weights],
        "wᵢ·f(xᵢ)": [f"{p:.6f}" for p in wf]
    })
    df.loc[len(df.index)] = ["—", "—", "Tổng:", f"{total:.6f}"]
    return df

if method in ["Hình thang", "Cả hai"]:
    st.markdown("**Phương pháp Hình thang**")
    X_trap = np.linspace(a, b, n_used_trap + 1)
    Y_trap = f_lambda(X_trap)
    st.dataframe(make_table(X_trap, Y_trap, "Hình thang"), use_container_width=True)

if method in ["Simpson", "Cả hai"]:
    st.markdown("**Phương pháp Simpson**")
    X_simp = np.linspace(a, b, n_used_simp + 1)
    Y_simp = f_lambda(X_simp)
    st.dataframe(make_table(X_simp, Y_simp, "Simpson"), use_container_width=True)

# TÙY CHỌN HIỂN THỊ ĐỒ THỊ
    
st.subheader("Tùy chọn hiển thị đồ thị")
fill_toggle = st.checkbox("Hiển thị vùng tô dưới đồ thị (tích phân)", value=True)

xx = np.linspace(a, b, 400)
yy = f_lambda(xx)

# ĐỒ THỊ MINH HỌA PHƯƠNG PHÁP HÌNH THANG

if method in ["Hình thang", "Cả hai"]:
    st.subheader("Minh họa phương pháp Hình thang")
    X_trap = np.linspace(a, b, n_used_trap + 1)
    Y_trap = f_lambda(X_trap)

    fig_trap = go.Figure()
    fig_trap.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
    if fill_toggle:
        for i in range(n_used_trap):
            xs = [X_trap[i], X_trap[i], X_trap[i+1], X_trap[i+1]]
            ys = [0, Y_trap[i], Y_trap[i+1], 0]
            fig_trap.add_trace(go.Scatter(
                x=xs, y=ys, fill="toself", fillcolor="rgba(255,0,0,0.3)",
                line=dict(color="rgba(255,0,0,0.2)"), showlegend=False))
    fig_trap.add_trace(go.Scatter(x=X_trap, y=Y_trap, mode="lines+markers",
                                  name="Các điểm chia", line=dict(color="red", dash="dot")))
    fig_trap.update_layout(xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig_trap, use_container_width=True)

# ĐỒ THỊ MINH HỌA PHƯƠNG PHÁP SIMPSON

if method in ["Simpson", "Cả hai"]:
    st.subheader("Minh họa phương pháp Simpson")
    X_simp = np.linspace(a, b, n_used_simp + 1)
    Y_simp = f_lambda(X_simp)

    fig_simp = go.Figure()
    fig_simp.add_trace(go.Scatter(
        x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))

    if fill_toggle:
        for i in range(0, n_used_simp, 2):
            xs = np.linspace(X_simp[i], X_simp[i+2], 40)
            coeffs = np.polyfit([X_simp[i], X_simp[i+1], X_simp[i+2]],
                                [Y_simp[i], Y_simp[i+1], Y_simp[i+2]], 2)
            ys = np.polyval(coeffs, xs)

            # Vùng tô
            fig_simp.add_trace(go.Scatter(
                x=[*xs, xs[-1], xs[0]], y=[*ys, 0, 0],
                fill="toself", fillcolor="rgba(0,255,0,0.25)",
                line=dict(color="rgba(0,255,0,0.2)"), showlegend=False))
            
            # Cung parabol nội suy 
            fig_simp.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name="Cung parabol nội suy" if i == 0 else None,
                line=dict(color="mediumseagreen", dash="dashdot"),
                showlegend=(i == 0)))

    # Các điểm chia
    fig_simp.add_trace(go.Scatter(
        x=X_simp, y=Y_simp, mode="lines+markers",
        name="Các điểm chia", line=dict(color="green", dash="dot")))

    fig_simp.update_layout(
        xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig_simp, use_container_width=True)
