import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="So sánh Hình thang và Simpson", layout="wide")

st.title("Hai phương pháp tính gần đúng tích phân")
st.markdown("### So sánh công thức Hình thang và Simpson")

# -------------------------------
# Chuẩn hóa hàm người dùng nhập
# -------------------------------
def normalize_expr(expr_str):
    expr_str = expr_str.lower()
    replacements = {'^': '**', 'ln': 'log', '√': 'sqrt', 'e': 'E'}
    for k, v in replacements.items():
        expr_str = expr_str.replace(k, v)
    return expr_str

# -------------------------------
# Các công thức tính tích phân gần đúng
# -------------------------------
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

# -------------------------------
# Giao diện nhập liệu
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    expr_str = st.text_input("Nhập hàm f(x):", "x**2")
    expr_str = normalize_expr(expr_str)
    a = st.number_input("Cận dưới a:", value=0.0)
    b = st.number_input("Cận trên b:", value=1.0)
with col2:
    mode = st.radio("Chọn cách nhập:", ["Nhập số khoảng n", "Nhập sai số ε"])
    if mode == "Nhập số khoảng n":
        n = st.number_input("Số khoảng chia n:", min_value=2, value=10, step=1)
        epsilon = None
    else:
        epsilon = st.number_input("Sai số ε:", min_value=1e-8, value=1e-4, format="%.1e")
        n = None

# -------------------------------
# Xử lý hàm f(x)
# -------------------------------
x = sp.Symbol('x')
try:
    f_expr = sp.sympify(expr_str)
    f_lambda = sp.lambdify(x, f_expr, "numpy")
    I_exact = float(sp.integrate(f_expr, (x, a, b)))
except Exception as e:
    st.error(f"Lỗi khi đọc hàm: {e}")
    st.stop()

# -------------------------------
# Hàm tính theo sai số hoặc số khoảng
# -------------------------------
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

# -------------------------------
# Tính kết quả
# -------------------------------
I_trap, n_used_trap = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, n)
I_simp, n_used_simp = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, n)
err_trap = abs(I_trap - I_exact)
err_simp = abs(I_simp - I_exact)

# -------------------------------
# Hiển thị kết quả
# -------------------------------
st.subheader("Kết quả so sánh")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.6f}")
cols[1].metric("Hình thang", f"{I_trap:.6f}", f"Sai số: {err_trap:.6f}")
cols[2].metric("Simpson", f"{I_simp:.6f}", f"Sai số: {err_simp:.6f}")
st.caption(f"Số khoảng: Hình thang = {n_used_trap}, Simpson = {n_used_simp}")

# -------------------------------
# Đồ thị vùng tích phân
# -------------------------------
st.subheader("Minh họa vùng tích phân")

xx = np.linspace(a, b, 400)
yy = f_lambda(xx)

fig = go.Figure()

# f(x)
fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="black", width=2)))

# --- Hình thang ---
X_trap = np.linspace(a, b, n_used_trap + 1)
Y_trap = f_lambda(X_trap)
for i in range(n_used_trap):
    xs = [X_trap[i], X_trap[i], X_trap[i+1], X_trap[i+1]]
    ys = [0, Y_trap[i], Y_trap[i+1], 0]
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(255,0,0,0.25)",
                             line=dict(color="rgba(255,0,0,0.2)"), showlegend=False))
fig.add_trace(go.Scatter(x=X_trap, y=Y_trap, mode="lines+markers", name="Hình thang",
                         line=dict(color="red", dash="dot"), marker=dict(size=6)))

# --- Simpson ---
X_simp = np.linspace(a, b, n_used_simp + 1)
Y_simp = f_lambda(X_simp)
for i in range(0, n_used_simp, 2):
    xs = np.linspace(X_simp[i], X_simp[i+2], 30)
    coeffs = np.polyfit([X_simp[i], X_simp[i+1], X_simp[i+2]],
                        [Y_simp[i], Y_simp[i+1], Y_simp[i+2]], 2)
    ys = np.polyval(coeffs, xs)
    fig.add_trace(go.Scatter(x=[*xs, xs[-1], xs[0]], y=[*ys, 0, 0],
                             fill="toself", fillcolor="rgba(0,255,0,0.25)",
                             line=dict(color="rgba(0,255,0,0.2)"), showlegend=False))
fig.add_trace(go.Scatter(x=X_simp, y=Y_simp, mode="lines+markers", name="Simpson",
                         line=dict(color="green", dash="dot"), marker=dict(size=6)))

fig.update_layout(
    xaxis_title="x", yaxis_title="f(x)",
    title="Vùng tích phân: Hình thang (đỏ) và Simpson (xanh)",
    height=500, legend_title="Phương pháp"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Bảng giá trị các điểm chia
# -------------------------------
st.subheader("Bảng giá trị tại các điểm chia")

colA, colB = st.columns(2)

with colA:
    st.markdown("**Phương pháp Hình thang**")
    df_trap = pd.DataFrame({
        "x_i": X_trap,
        "f(x_i)": [f"{val:.6f}" for val in Y_trap]
    })
    st.dataframe(df_trap, use_container_width=True)

with colB:
    st.markdown("**Phương pháp Simpson**")
    df_simp = pd.DataFrame({
        "x_i": X_simp,
        "f(x_i)": [f"{val:.6f}" for val in Y_simp]
    })
    st.dataframe(df_simp, use_container_width=True)

# -------------------------------
# Biểu đồ hội tụ sai số
# -------------------------------
st.subheader("Biểu đồ hội tụ sai số")

ns = [4, 8, 16, 32, 64, 128]
err_trap_list = [abs(trapezoidal_rule(f_lambda, a, b, ni) - I_exact) for ni in ns]
err_simp_list = [abs(simpson_rule(f_lambda, a, b, ni) - I_exact) for ni in ns]

fig_err = go.Figure()
fig_err.add_trace(go.Scatter(x=np.log10(ns), y=np.log10(err_trap_list),
                             mode="lines+markers", name="Hình thang", line=dict(color="red")))
fig_err.add_trace(go.Scatter(x=np.log10(ns), y=np.log10(err_simp_list),
                             mode="lines+markers", name="Simpson", line=dict(color="green")))

fig_err.update_layout(
    xaxis_title="log10(n)", yaxis_title="log10(|Sai số|)",
    title="So sánh tốc độ hội tụ sai số",
    height=500, legend_title="Công thức"
)
st.plotly_chart(fig_err, use_container_width=True)
