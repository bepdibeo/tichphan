import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="So sánh Hình thang và Simpson", layout="wide")

st.title("Hai phương pháp tính gần đúng tích phân")
st.markdown("### Công thức Hình thang và Simpson")

# Chuẩn hóa hàm nhập
def normalize_expr(expr_str):
    expr_str = expr_str.lower()
    replacements = {'^': '**', 'ln': 'log', '√': 'sqrt', 'e': 'E'}
    for k, v in replacements.items():
        expr_str = expr_str.replace(k, v)
    return expr_str

# Công thức Hình thang và Simpson
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
    return (h/3) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))

# --- Nhập liệu ---
col1, col2 = st.columns(2)
with col1:
    expr_str = st.text_input("Nhập hàm f(x):", "exp(x)")
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

# --- Chuẩn bị hàm ---
x = sp.Symbol('x')
try:
    f_expr = sp.sympify(expr_str)
    f_lambda = sp.lambdify(x, f_expr, "numpy")
    I_exact = float(sp.integrate(f_expr, (x, a, b)))
except Exception as e:
    st.error(f"Lỗi khi đọc hàm: {e}")
    st.stop()

# --- Hàm tính theo sai số ---
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

# --- Tính kết quả ---
I_trap, n_used_trap = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, n)
I_simp, n_used_simp = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, n)
err_trap = abs(I_trap - I_exact)
err_simp = abs(I_simp - I_exact)

# --- Hiển thị kết quả ---
st.subheader("Kết quả so sánh")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.6f}")
cols[1].metric("Hình thang", f"{I_trap:.6f}", f"Sai số: {err_trap:.6f}")
cols[2].metric("Simpson", f"{I_simp:.6f}", f"Sai số: {err_simp:.6f}")
st.caption(f"Số khoảng: Hình thang = {n_used_trap}, Simpson = {n_used_simp}")

# --- Minh họa vùng tích phân ---
st.subheader("Minh họa vùng tích phân (tách biệt)")

xx = np.linspace(a, b, 400)
yy = f_lambda(xx)

colA, colB = st.columns(2)

# Biểu đồ Hình thang
with colA:
    fig_trap = go.Figure()
    fig_trap.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
    X_trap = np.linspace(a, b, n_used_trap + 1)
    Y_trap = f_lambda(X_trap)
    fig_trap.add_trace(go.Scatter(x=X_trap, y=Y_trap, mode="markers+lines",
                                  name=f"Hình thang (n={n_used_trap})",
                                  line=dict(color="red", dash="dot")))
    fig_trap.update_layout(title="Phương pháp Hình thang", xaxis_title="x", yaxis_title="f(x)", height=400)
    st.plotly_chart(fig_trap, use_container_width=True)

# Biểu đồ Simpson
with colB:
    fig_simp = go.Figure()
    fig_simp.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
    X_simp = np.linspace(a, b, n_used_simp + 1)
    Y_simp = f_lambda(X_simp)
    fig_simp.add_trace(go.Scatter(x=X_simp, y=Y_simp, mode="markers+lines",
                                  name=f"Simpson (n={n_used_simp})",
                                  line=dict(color="green", dash="dot")))
    fig_simp.update_layout(title="Phương pháp Simpson", xaxis_title="x", yaxis_title="f(x)", height=400)
    st.plotly_chart(fig_simp, use_container_width=True)

# --- Bảng giá trị các điểm chia ---
st.subheader("Bảng giá trị tại các điểm chia")

df_trap = pd.DataFrame({"x_i": X_trap, "f(x_i)": [f"{v:.6f}" for v in Y_trap]})
df_simp = pd.DataFrame({"x_i": X_simp, "f(x_i)": [f"{v:.6f}" for v in Y_simp]})

colT, colS = st.columns(2)
with colT:
    st.markdown("**Phương pháp Hình thang**")
    st.dataframe(df_trap, use_container_width=True)
with colS:
    st.markdown("**Phương pháp Simpson**")
    st.dataframe(df_simp, use_container_width=True)

# --- Biểu đồ hội tụ sai số ---
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
    xaxis_title="log10(n)",
    yaxis_title="log10(|Sai số|)",
    title="So sánh tốc độ hội tụ sai số (trên thang log-log)",
    height=500,
    legend_title="Công thức"
)
st.plotly_chart(fig_err, use_container_width=True)
