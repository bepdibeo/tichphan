import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Tích phân gần đúng", layout="wide")
st.title("Hai phương pháp tính gần đúng tích phân")
st.markdown("### Phương pháp Hình thang và Simpson")

#  Chuẩn hóa & hàm hỗ trợ 
def normalize_expr(expr_str):
    return expr_str.strip().replace('^', '**').replace('ln', 'log').replace('√', 'sqrt').replace('π', 'pi')

def make_vectorized(f_lambda):  # 
    return lambda x_arr: np.full_like(x_arr, float(f_lambda(0))) if np.isscalar(f_lambda(0)) else np.asarray(f_lambda(x_arr), dtype=float)

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1) 
    y = f(x) 
    h = (b - a) / n
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

def simpson_rule(f, a, b, n):
    if n % 2: n += 1
    x = np.linspace(a, b, n + 1) 
    y = f(x) 
    h = (b - a) / n
    return (h/3)*(y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))

#  Giao diện nhập 
col1, col2 = st.columns(2)
with col1:
    expr_input = st.text_input("Nhập hàm f(x):", "x**2")
    a = st.number_input("Cận dưới a:", value=0.0, min_value=-np.inf)
    b = st.number_input("Cận trên b:", value=1.0, min_value=-np.inf)

with col2:
    method = st.radio("Chọn phương pháp:", ["Hình thang", "Simpson", "Cả hai"])
    mode = st.radio("Chọn cách nhập:", ["Nhập số khoảng n", "Nhập sai số ε"])
    if mode == "Nhập số khoảng n":
        n_input, epsilon = st.number_input("Số khoảng chia n:", 1, step=1, value=10), None
    else:
        epsilon, n_input = st.number_input("Sai số ε:", min_value=1e-12, value=1e-4, format="%.1e"), None

#  Tạo hàm 
x = sp.Symbol('x')
expr_str = normalize_expr(expr_input)
try:
    f_expr = sp.sympify(expr_str, locals={'e': sp.E, 'pi': sp.pi})
    is_const = not f_expr.free_symbols
    const_val = float(f_expr) if is_const else None
    f_lambda = (lambda t: np.full_like(t, const_val)) if is_const else sp.lambdify(x, f_expr, "numpy")
except Exception:
    st.error("Cú pháp hàm không hợp lệ. Ví dụ: sin(x), exp(x), x**2, log(x), ..."); st.stop()

if b <= a: st.error("Cận trên b phải lớn hơn cận dưới a."); st.stop()

#  Tích phân chính xác
try:
    I_exact = float(f_expr * (b - a)) if is_const else float(sp.integrate(f_expr, (x, a, b)))
except Exception:
    I_exact = None

# Kiểm tra I_exact có hợp lệ không
if I_exact is None or not np.isfinite(I_exact):
    st.subheader("Kết quả")
    st.metric("Tích phân chính xác", "—" if I_exact is None else str(I_exact))
    st.warning("Tích phân chính xác không tính được hoặc lỗi số học. Không tính tích phân gần đúng.")
    st.stop()  # dừng app, không đi tiếp phần bảng giá trị hay đồ thị

#  Hàm tính 
def compute_with_tolerance(f, a, b, rule, epsilon=None, n=None):
    prev, n = None, 2 if epsilon else n
    while True:
        I = rule(f, a, b, n)
        if epsilon is None or (prev and abs(I - prev) < epsilon): return I, n
        prev, n = I, n*2
        if n > 1e7: raise RuntimeError("Không hội tụ. Hãy tăng ε.")

def theoretical_error(f_expr, a, b, n, method):
    try:
        grid = np.linspace(a, b, 1000)
        if method == "Hình thang":
            f2 = sp.diff(f_expr, x, 2)
            f2_func = sp.lambdify(x, f2, "numpy")
            vals = np.abs(f2_func(grid))
            vals = vals[np.isfinite(vals)]  # loại bỏ NaN, inf
            if len(vals) == 0:
                return None
            M2 = np.max(vals)
            return ((b - a)**3) / (12 * n**2) * M2
        elif method == "Simpson":
            f4 = sp.diff(f_expr, x, 4)
            f4_func = sp.lambdify(x, f4, "numpy")
            vals = np.abs(f4_func(grid))
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return None
            M4 = np.max(vals)
            return ((b - a)**5) / (180 * n**4) * M4
    except Exception:
        return None

#  Tính toán 
n_user = int(max(1, n_input or 1))
if method in ["Simpson", "Cả hai"] and n_user % 2: st.warning(f"Simpson yêu cầu n chẵn — đã đổi {n_user}→{n_user+1}"); n_user += 1
I_trap = I_simp = err_trap = err_simp = None

if method in ["Hình thang", "Cả hai"]:
    I_trap, n_t = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, n_user)
    err_trap = abs(I_trap - I_exact) if I_exact else None
if method in ["Simpson", "Cả hai"]:
    I_simp, n_s = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, n_user)
    err_simp = abs(I_simp - I_exact) if I_exact else None

#  Hiển thị kết quả 
st.subheader("Kết quả")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.6f}" if I_exact else "—")

def show_result(col, title, I, n, err, method):
    e_theory = theoretical_error(f_expr, a, b, n, method)
    note = (f"Sai số [Thực: {err:.3g}]" if err else "") + (f" [Lý thuyết: {e_theory:.3g}]" if e_theory else "")
    col.metric(f"{title} (n={n})", f"{I:.6f}", note)

if I_trap is not None: show_result(cols[1], "Hình thang", I_trap, n_t, err_trap, "Hình thang")
if I_simp is not None: show_result(cols[2], "Simpson", I_simp, n_s, err_simp, "Simpson")

#  Bảng giá trị 
st.subheader("Bảng giá trị chi tiết")
def make_table(xv, yv, w, h, title, coef):
    df = pd.DataFrame({"i": range(len(xv)), "x_i": xv, "f(x_i)": yv, "Trọng số": w, "Trọng số × f(x_i)": w*yv})
    st.markdown(f"#### {title}")
    st.dataframe(df.style.format({"x_i": "{:.6f}", "f(x_i)": "{:.6f}", "Trọng số × f(x_i)": "{:.6f}"}), use_container_width=True)
    st.latex(rf"\sum w_i f(x_i) = {np.sum(w*yv):.6f},\ I \approx {coef} \times {np.sum(w*yv):.6f} = {h*np.sum(w*yv):.6f}")

if method in ["Hình thang", "Cả hai"]:
    X = np.linspace(a, b, n_t + 1); Y = f_lambda(X)
    W = np.ones_like(X); W[1:-1] = 2; h = (b - a) / n_t
    make_table(X, Y, W, h/2, "Phương pháp Hình thang", "h/2")
if method in ["Simpson", "Cả hai"]:
    X = np.linspace(a, b, n_s + 1); Y = f_lambda(X)
    W = np.array([1 if i in [0,len(X)-1] else 4 if i%2 else 2 for i in range(len(X))])
    h = (b - a) / n_s
    make_table(X, Y, W, h/3, "Phương pháp Simpson (1/3)", "h/3")

# Đồ thị 
st.subheader("Tùy chọn hiển thị đồ thị")
fill = st.checkbox("Hiển thị vùng tô tích phân", True)
smooth = st.slider("Độ mịn cung parabol (Simpson)", 10, 400, 80, 10)
xx, yy = np.linspace(a, b, 800), f_lambda(np.linspace(a, b, 800))

def plot_area(method, X, Y, fillcolor, curvecolor, interp=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
    if method == "Hình thang":
        for i in range(len(X)-1):
            xs, ys = [X[i], X[i], X[i+1], X[i+1]], [0, Y[i], Y[i+1], 0]
            if fill: fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=fillcolor, line=dict(color=fillcolor), showlegend=False))
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines+markers", name="Các điểm chia", line=dict(color="red", dash="dot")))
    else:  # Simpson
        for i in range(0, len(X)-2, 2):
            xs = np.linspace(X[i], X[i+2], smooth)
            ys = np.polyval(np.polyfit([X[i], X[i+1], X[i+2]], [Y[i], Y[i+1], Y[i+2]], 2), xs)
            if fill:
                fig.add_trace(go.Scatter(x=np.concatenate((xs,[xs[-1],xs[0]])), y=np.concatenate((ys,[0,0])),
                                         fill="toself", fillcolor=fillcolor, line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Cung parabol nội suy" if i==0 else None,
                                     line=dict(color=curvecolor, dash="dashdot"), showlegend=(i==0)))
        fig.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="Các điểm chia", line=dict(color="red", dash="dot")))
    fig.update_layout(xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig, use_container_width=True)

if method in ["Hình thang", "Cả hai"]: 
    st.subheader("Minh họa phương pháp Hình thang")
    plot_area("Hình thang", np.linspace(a, b, n_t + 1), f_lambda(np.linspace(a, b, n_t + 1)), "rgba(255,0,0,0.1)", "red")
if method in ["Simpson", "Cả hai"]:
    st.subheader("Minh họa phương pháp Simpson")
    plot_area("Simpson", np.linspace(a, b, n_s + 1), f_lambda(np.linspace(a, b, n_s + 1)), "rgba(255,215,0,0.1)", "gold")

