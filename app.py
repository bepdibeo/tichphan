import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Tích phân gần đúng", layout="wide")
st.title("Hai phương pháp tính gần đúng tích phân")
st.markdown("### Phương pháp Hình thang và Simpson")

# Chuẩn hóa biểu thức nhập từ người dùng 
def normalize_expr(expr_str):
    expr_str = expr_str.strip().replace('^', '**').replace('ln', 'log').replace('√', 'sqrt').replace('π', 'pi')
    return expr_str

# Đảm bảo trả về mảng float các điểm chia
def make_vectorized(f_lambda):
    def f_vec(x_arr):
        y = np.asarray(f_lambda(x_arr))
        if y.shape == ():
            val = float(y)
            return np.full_like(x_arr, val, dtype=float)
        return y.astype(float)
    return f_vec

# Hai công thức tích phân 
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

# Giao diện nhập
col1, col2 = st.columns(2)
with col1:
    expr_input = st.text_input("Nhập hàm f(x):", "x**2")
    a = st.number_input("Cận dưới a:", value=0.0)
    b = st.number_input("Cận trên b:", value=1.0)
with col2:
    method = st.radio("Chọn phương pháp:", ["Hình thang", "Simpson", "Cả hai"])
    mode = st.radio("Chọn cách nhập:", ["Nhập số khoảng n", "Nhập sai số ε"])
    if mode == "Nhập số khoảng n":
        n_input = st.number_input("Số khoảng chia n:", min_value=1, value=10, step=1)
        epsilon = None
    else:
        epsilon = st.number_input("Sai số ε:", min_value=1e-12, value=1e-4, format="%.1e")
        n_input = None

# Tạo hàm từ biểu thức 
x = sp.Symbol('x')
expr_str = normalize_expr(expr_input)

try:
    f_expr = sp.sympify(expr_str, locals={'e': sp.E, 'pi': sp.pi})
    is_constant = not f_expr.free_symbols
    if is_constant:
        const_val = float(f_expr)
        f_lambda = lambda t: np.full_like(t, const_val)
    else:
        f_lambda = sp.lambdify(x, f_expr, "numpy")
except Exception:
    st.error("Cú pháp hàm không hợp lệ. Ví dụ: sin(x), exp(x), x**2, log(x), ...")
    st.stop()

# Kiểm tra miền xác định 
if b <= a:
    st.error("Cận trên b phải lớn hơn cận dưới a.")
    st.stop()

X_test = np.linspace(a, b, 400)
try:
    Y_test = f_lambda(X_test)
except Exception as e:
    st.error(f"Lỗi khi tính giá trị hàm trên đoạn [{a}, {b}]. Chi tiết: {e}")
    st.stop()

if np.any(np.iscomplex(Y_test)) or np.any(~np.isfinite(Y_test)):
    st.error("Hàm có giá trị phức, vô hạn hoặc NaN trong khoảng tích phân.")
    st.stop()

# Tích phân chính xác (nếu có)
try:
    I_exact = float(f_expr * (b - a)) if is_constant else float(sp.integrate(f_expr, (x, a, b)))
except Exception:
    I_exact = None
    st.warning("Không thể tính tích phân chính xác. Sẽ chỉ tính gần đúng.")

# Hàm tính tích phân theo n hoặc epsilon
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
            if n > 10**7:
                raise RuntimeError("Không hội tụ. Hãy tăng ε.")
    else:
        I1 = rule_func(f, a, b, n)
    return I1, n

# Hàm ước lượng sai số
def theoretical_error(f_expr, a, b, n, method):
    try:
        if method == "Hình thang":
            f2 = sp.diff(f_expr, x, 2)
            f2_vec = make_vectorized(sp.lambdify(x, f2, "numpy"))
            M2 = np.max(np.abs(f2_vec(np.linspace(a, b, 1000))))
            return ((b - a)**3) / (12 * n**2) * M2
        elif method == "Simpson":
            f4 = sp.diff(f_expr, x, 4)
            f4_vec = make_vectorized(sp.lambdify(x, f4, "numpy"))
            M4 = np.max(np.abs(f4_vec(np.linspace(a, b, 1000))))
            return ((b - a)**5) / (180 * n**4) * M4
    except Exception:
        return None

# Tính toán 
n_user = int(max(1, int(n_input))) if mode == "Nhập số khoảng n" else None

if method in ["Simpson", "Cả hai"] and mode == "Nhập số khoảng n" and n_user % 2 == 1:
    st.warning(f"Simpson yêu cầu n chẵn — đã đổi từ {n_user} thành {n_user + 1}")
    n_user += 1

I_trap = I_simp = err_trap = err_simp = None
if method in ["Hình thang", "Cả hai"]:
    I_trap, n_t = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, n_user)
    err_trap = abs(I_trap - I_exact) if I_exact is not None else None

if method in ["Simpson", "Cả hai"]:
    I_simp, n_s = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, n_user)
    err_simp = abs(I_simp - I_exact) if I_exact is not None else None

# Hiển thị kết quả 
st.subheader("Kết quả")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.6f}" if I_exact is not None else "—")

if I_trap is not None:
    e_trap_theory = theoretical_error(f_expr, a, b, n_t, "Hình thang")
    cols[1].metric(f"Hình thang (n={n_t})", f"{I_trap:.6f}", f"Sai số: {err_trap:.3g}" if err_trap else "")

if I_simp is not None:
    e_simp_theory = theoretical_error(f_expr, a, b, n_s, "Simpson")
    cols[2].metric(f"Simpson (n={n_s})", f"{I_simp:.6f}", f"Sai số: {err_simp:.3g}" if err_simp else "")


# So sánh chuyển lên ngay sau kết quả
if method == "Cả hai" and I_trap is not None and I_simp is not None:
    st.markdown("### So sánh hai phương pháp")

    diff_abs = abs(I_simp - I_trap)
    diff_percent = (diff_abs / abs(I_simp)) * 100 if I_simp != 0 else None

    if I_exact is not None:
        err_trap_exact = abs(I_trap - I_exact)
        err_simp_exact = abs(I_simp - I_exact)
        better = "Simpson" if err_simp_exact < err_trap_exact else "Hình thang"
        st.success(
            f"- **Phương pháp {better} cho độ chính xác cao hơn.**  "
            f"(Sai số Simpson = {err_simp_exact:.3g}, Hình thang = {err_trap_exact:.3g})"
        )
    else:
        st.info(
            f"- Chênh lệch tuyệt đối giữa hai phương pháp: {diff_abs:.6e}  \n"
            f"- Sai khác tương đối: {diff_percent:.3f}%"
        )


# Hiển thị bảng giá trị chi tiết 
st.subheader("Bảng giá trị chi tiết cho từng phương pháp")

def make_table_with_formula(x_vals, y_vals, weights, h, title, coef_text, coef_display):
    weighted_fx = weights * y_vals
    df = pd.DataFrame({
        "i": np.arange(len(x_vals)),
        "x_i": x_vals,
        "f(x_i)": y_vals,
        "Trọng số": weights,
        "Trọng số × f(x_i)": weighted_fx
    })
    total_sum = weighted_fx.sum()
    result = h * total_sum

    st.markdown(f"#### {title}")
    st.dataframe(
        df.style.format({
            "x_i": "{:.6f}",
            "f(x_i)": "{:.6f}",
            "Trọng số": "{:.0f}",
            "Trọng số × f(x_i)": "{:.6f}"
        }),
        use_container_width=True
    )

    st.latex(
        rf"""
        \begin{{aligned}}
        \text{{Tổng: }} & \sum w_i f(x_i) = {total_sum:.6f} \\
        I &\approx {coef_display} \times {total_sum:.6f} = {result:.6f}
        \end{{aligned}}
        """
    )
    return result


# Hình thang
I_trap_table = None
if method in ["Hình thang", "Cả hai"]:
    X_trap = np.linspace(a, b, n_t + 1)
    Y_trap = f_lambda(X_trap)
    W_trap = np.ones(len(X_trap))
    W_trap[0] = W_trap[-1] = 1
    W_trap[1:-1] = 2
    h_trap = (b - a) / n_t
    I_trap_table = make_table_with_formula(X_trap, Y_trap, W_trap, h_trap / 2, 
                                           "Phương pháp Hình thang", "h/2", r"\frac{h}{2}")

# Simpson 
I_simp_table = None
if method in ["Simpson", "Cả hai"]:
    X_simp = np.linspace(a, b, n_s + 1)
    Y_simp = f_lambda(X_simp)
    W_simp = np.ones(len(X_simp))
    for i in range(1, len(W_simp) - 1):
        W_simp[i] = 4 if i % 2 == 1 else 2
    h_simp = (b - a) / n_s
    I_simp_table = make_table_with_formula(X_simp, Y_simp, W_simp, h_simp / 3, 
                                           "Phương pháp Simpson (1/3)", "h/3", r"\frac{h}{3}")


# Tùy chọn đồ thị
st.subheader("Tùy chọn hiển thị đồ thị")
fill_toggle = st.checkbox("Hiển thị vùng tô dưới đồ thị (tích phân)", value=True)
interp_points = st.slider("Độ mịn của cung parabol (Simpson)", min_value=10, max_value=400, value=80, step=10)

xx = np.linspace(a, b, 800)
yy = f_lambda(xx)

# Hình thang 
if method in ["Hình thang", "Cả hai"]:
    st.subheader("Minh họa phương pháp Hình thang")
    X = np.linspace(a, b, n_t + 1)
    Y = f_lambda(X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
    if fill_toggle:
        for i in range(n_t):
            xs = [X[i], X[i], X[i+1], X[i+1]]
            ys = [0.0, Y[i], Y[i+1], 0.0]
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(255, 0, 0, 0.15)", line=dict(color="rgba(255,0,0,0.2)"), showlegend=False))
    fig.add_trace(go.Scatter(x=X, y=Y, mode="lines+markers", name="Các điểm chia", line=dict(color="red", dash="dot")))
    fig.update_layout(xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig, use_container_width=True)

# Simpson 
if method in ["Simpson", "Cả hai"]:
    st.subheader("Minh họa phương pháp Simpson")
    X = np.linspace(a, b, n_s + 1)
    Y = f_lambda(X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))

    for i in range(0, len(X)-2, 2):
        xs = np.linspace(X[i], X[i+2], interp_points)
        coeffs = np.polyfit([X[i], X[i+1], X[i+2]], [Y[i], Y[i+1], Y[i+2]], 2)
        ys = np.polyval(coeffs, xs)
        if fill_toggle:
            fig.add_trace(go.Scatter(
                x=np.concatenate((xs, [xs[-1], xs[0]])),
                y=np.concatenate((ys, [0.0, 0.0])),
                fill="toself",
                fillcolor="rgba(255, 215, 0, 0.15)",
                line=dict(color="rgba(255,0,0,0.2)"),
                showlegend=False))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name="Cung parabol nội suy" if i==0 else None,
            line=dict(color="gold", dash="dashdot"),
            showlegend=(i==0)))
    fig.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="Các điểm chia", line=dict(color="red", dash="dot")))
    fig.update_layout(xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig, use_container_width=True)

