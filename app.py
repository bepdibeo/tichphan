import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="So sánh Hình thang và Simpson", layout="wide")

st.title("Hai phương pháp tính gần đúng tích phân")
st.markdown("### Công thức Hình thang và Simpson – minh họa trực quan")

# ====== Hàm tiện ích ======
def normalize_expr(expr_str):
    expr_str = expr_str.lower()
    replacements = {'^': '**', 'ln': 'log', '√': 'sqrt', 'e': 'E'}
    for k, v in replacements.items():
        expr_str = expr_str.replace(k, v)
    return expr_str

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

# ====== Giao diện nhập ======
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
        epsilon = st.number_input("Sai số ε:", min_value=1e-8, value=1e-4, format="%.6f")
        n = None

# ====== Chuẩn bị hàm ======
x = sp.Symbol('x')
try:
    f_expr = sp.sympify(expr_str)
    f_lambda = sp.lambdify(x, f_expr, "numpy")
    I_exact = float(sp.integrate(f_expr, (x, a, b)))
except Exception as e:
    st.error(f"Lỗi khi đọc hàm: {e}")
    st.stop()

# ====== Hàm tính có ε ======
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

# ====== Tính toán ======
I_trap, n_used_trap = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, n)
I_simp, n_used_simp = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, n)
err_trap = abs(I_trap - I_exact)
err_simp = abs(I_simp - I_exact)

# ====== Kết quả ======
st.subheader("Kết quả so sánh")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.8f}")
cols[1].metric("Hình thang", f"{I_trap:.8f}", f"Sai số: {err_trap:.8f}")
cols[2].metric("Simpson", f"{I_simp:.8f}", f"Sai số: {err_simp:.8f}")
st.caption(f"Số khoảng: Hình thang = {n_used_trap}, Simpson = {n_used_simp}")

# ====== Biểu đồ minh họa riêng ======
st.subheader("Minh họa vùng tích phân")

colA, colB = st.columns(2)

xx = np.linspace(a, b, 400)
yy = f_lambda(xx)

# --- Biểu đồ Hình thang ---
with colA:
    st.markdown("#### Phương pháp Hình thang")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))

    X_trap = np.linspace(a, b, n_used_trap + 1)
    Y_trap = f_lambda(X_trap)
    for i in range(len(X_trap) - 1):
        x_fill = [X_trap[i], X_trap[i+1], X_trap[i+1], X_trap[i]]
        y_fill = [0, 0, Y_trap[i+1], Y_trap[i]]
        fig1.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself",
                                 fillcolor="rgba(255,100,100,0.3)",
                                 line=dict(color="rgba(255,100,100,0.3)"),
                                 showlegend=False))
    fig1.add_trace(go.Scatter(x=X_trap, y=Y_trap, mode="markers+lines",
                             name=f"Hình thang (n={n_used_trap})",
                             line=dict(color="red", dash="dot"), marker=dict(size=6)))
    fig1.update_layout(height=500, title="Phương pháp Hình thang", xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig1, use_container_width=True)

    df_trap = pd.DataFrame({"i": range(len(X_trap)), "xᵢ": X_trap, "f(xᵢ)": Y_trap})
    df_trap["xᵢ"] = df_trap["xᵢ"].map(lambda v: f"{v:.6f}")
    df_trap["f(xᵢ)"] = df_trap["f(xᵢ)"].map(lambda v: f"{v:.6f}")
    st.dataframe(df_trap, hide_index=True, use_container_width=True)

# --- Biểu đồ Simpson ---
with colB:
    st.markdown("#### Phương pháp Simpson")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))

    X_simp = np.linspace(a, b, n_used_simp + 1)
    Y_simp = f_lambda(X_simp)
    for i in range(0, len(X_simp) - 2, 2):
        x_sub = np.linspace(X_simp[i], X_simp[i+2], 30)
        y_sub = np.interp(x_sub, [X_simp[i], X_simp[i+1], X_simp[i+2]],
                          [Y_simp[i], Y_simp[i+1], Y_simp[i+2]])
        fig2.add_trace(go.Scatter(x=x_sub, y=y_sub, fill="toself",
                                 fillcolor="rgba(100,255,100,0.3)",
                                 line=dict(color="green"), showlegend=False))
    fig2.add_trace(go.Scatter(x=X_simp, y=Y_simp, mode="markers+lines",
                             name=f"Simpson (n={n_used_simp})",
                             line=dict(color="green", dash="dot"), marker=dict(size=6)))
    fig2.update_layout(height=500, title="Phương pháp Simpson", xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig2, use_container_width=True)

    df_simp = pd.DataFrame({"i": range(len(X_simp)), "xᵢ": X_simp, "f(xᵢ)": Y_simp})
    df_simp["xᵢ"] = df_simp["xᵢ"].map(lambda v: f"{v:.6f}")
    df_simp["f(xᵢ)"] = df_simp["f(xᵢ)"].map(lambda v: f"{v:.6f}")
    st.dataframe(df_simp, hide_index=True, use_container_width=True)

# ====== Biểu đồ hội tụ sai số ======
st.subheader("Biểu đồ hội tụ sai số")

ns = np.array([4, 8, 16, 32, 64, 128])
err_trap_list = np.array([abs(trapezoidal_rule(f_lambda, a, b, ni) - I_exact) for ni in ns])
err_simp_list = np.array([abs(simpson_rule(f_lambda, a, b, ni) - I_exact) for ni in ns])

# log–log
x_log = np.log10(ns)
y_trap_log = np.log10(err_trap_list)
y_simp_log = np.log10(err_simp_list)

# đường hồi quy tuyến tính
fit_trap = np.polyfit(x_log, y_trap_log, 1)
fit_simp = np.polyfit(x_log, y_simp_log, 1)

fig_err = go.Figure()
fig_err.add_trace(go.Scatter(x=x_log, y=y_trap_log, mode="markers+lines",
                             name="Hình thang", line=dict(color="red")))
fig_err.add_trace(go.Scatter(x=x_log, y=y_simp_log, mode="markers+lines",
                             name="Simpson", line=dict(color="green")))

# vẽ đường thẳng hồi quy
fig_err.add_trace(go.Scatter(x=x_log, y=np.polyval(fit_trap, x_log),
                             mode="lines", name=f"Fit Hình thang (slope={fit_trap[0]:.2f})",
                             line=dict(color="red", dash="dot")))
fig_err.add_trace(go.Scatter(x=x_log, y=np.polyval(fit_simp, x_log),
                             mode="lines", name=f"Fit Simpson (slope={fit_simp[0]:.2f})",
                             line=dict(color="green", dash="dot")))

fig_err.update_layout(
    xaxis_title="log₁₀(n)",
    yaxis_title="log₁₀(|Sai số|)",
    title="So sánh tốc độ hội tụ sai số (log–log)",
    height=500,
    legend_title="Phương pháp"
)
st.plotly_chart(fig_err, use_container_width=True)
