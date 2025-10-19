import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

st.set_page_config(page_title="Tích phân gần đúng – Hình thang và Simpson", layout="wide")

st.title("ỨNG DỤNG MINH HỌA TÍNH TÍCH PHÂN GẦN ĐÚNG")
st.markdown("### Bằng công thức Hình thang và Simpson")

# Giao diện nhập 
col1, col2 = st.columns(2)
with col1:
    func_str = st.text_input("Hàm f(x):", "x**2 + 2*x + 1")
with col2:
    mode = st.radio("Chọn cách nhập:", ["Nhập số khoảng n", "Nhập sai số tối đa ε"])

a = st.number_input("Cận dưới a:", value=0.0)
b = st.number_input("Cận trên b:", value=2.0)

methods = st.multiselect(
    "Chọn công thức muốn sử dụng:",
    ["Hình thang", "Simpson"],
    default=["Hình thang", "Simpson"]
)

if mode == "Nhập số khoảng n":
    n = st.slider("Số khoảng chia n:", min_value=2, max_value=200, value=6, step=2)
    eps = None
else:
    eps = st.number_input("Sai số tối đa ε:", value=0.001, min_value=1e-8, step=1e-4, format="%.6f")
    n = None

# Chuẩn bị hàm
x = sp.Symbol("x")
try:
    f = sp.sympify(func_str)
    f_lambda = sp.lambdify(x, f, "numpy")
except Exception as e:
    st.error(f"Lỗi trong biểu thức hàm: {e}")
    st.stop()

# Tích phân chính xác
try:
    I_exact = float(sp.integrate(f, (x, a, b)))
except Exception:
    I_exact = None

# Hàm tính tích phân gần đúng
def calc_trapezoidal(f_lambda, a, b, n):
    X = np.linspace(a, b, n + 1)
    Y = f_lambda(X)
    h = (b - a) / n
    I = (h / 2) * (Y[0] + 2 * np.sum(Y[1:-1]) + Y[-1])
    return I, X, Y

def calc_simpson(f_lambda, a, b, n):
    if n % 2 == 1:
        n += 1
    X = np.linspace(a, b, n + 1)
    Y = f_lambda(X)
    h = (b - a) / n
    I = (h / 3) * (Y[0] + 4 * np.sum(Y[1:-1:2]) + 2 * np.sum(Y[2:-2:2]) + Y[-1])
    return I, X, Y

# Nếu nhập ε thì tìm n tự động
if eps is not None and I_exact is not None:
    f2 = sp.diff(f, (x, 2))
    f2_lambda = sp.lambdify(x, f2, "numpy")
    xs = np.linspace(a, b, 200)
    try:
        M2 = np.max(np.abs(f2_lambda(xs)))
        n_est = int(np.ceil(np.sqrt(((b - a) ** 3 * M2) / (12 * eps))))
        n = max(2, n_est)
        st.info(f"🔍 Ước lượng n tối ưu ≈ **{n}** (theo sai số hình thang ε = {eps})")
    except Exception:
        st.warning("Không thể ước lượng sai số do hàm f''(x) không khả dụng.")
        n = 10

# Tính tích phân gần đúng
results = {}
if "Hình thang" in methods:
    I_trap, X_trap, Y_trap = calc_trapezoidal(f_lambda, a, b, n)
    results["Hình thang"] = (I_trap, X_trap, Y_trap)
if "Simpson 1/3" in methods:
    I_simp, X_simp, Y_simp = calc_simpson(f_lambda, a, b, n)
    results["Simpson 1/3"] = (I_simp, X_simp, Y_simp)

# Hiển thị kết quả
st.subheader("Kết quả tính toán")
cols = st.columns(len(results) + 1)
i = 0
for name, (I, _, _) in results.items():
    cols[i].metric(name, f"{I:.6f}")
    i += 1
if I_exact is not None:
    cols[-1].metric("Tích phân chính xác", f"{I_exact:.6f}")

if I_exact is not None:
    for name, (I, _, _) in results.items():
        st.write(f"🔹 Sai số {name}: {abs(I - I_exact):.3e}")

# Bảng giá trị
st.subheader("Bảng giá trị (xi, f(xi))")
for name, (_, X, Y) in results.items():
    st.markdown(f"#### {name}")
    data = {"i": range(len(X)), "xi": X, "f(xi)": Y}
    st.dataframe(data, hide_index=True, use_container_width=True)

# Đồ thị tích phân gần đúng
st.subheader("Minh họa vùng tích phân")
xx = np.linspace(a, b, 400)
yy = f_lambda(xx)
fig = go.Figure()

fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=np.concatenate(([a], xx, [b])), y=np.concatenate(([0], yy, [0])),
                         fill='toself', name="Vùng tích phân", fillcolor="skyblue", opacity=0.3))

colors = {"Hình thang": "red", "Simpson": "green"}
for name, (_, X, Y) in results.items():
    fig.add_trace(go.Scatter(x=X, y=Y, mode="markers+lines",
                             name=f"Điểm {name} (n={n})",
                             line=dict(color=colors.get(name, "gray"), dash="dot")))

fig.update_layout(xaxis_title="x", yaxis_title="f(x)", height=500, legend_title="Chú thích")
st.plotly_chart(fig, use_container_width=True)

# Biểu đồ hội tụ sai số 
if I_exact is not None:
    st.subheader("Biểu đồ hội tụ sai số")
    ns = np.array([2, 4, 8, 16, 32, 64, 128])
    err_trap = []
    err_simp = []
    for n_ in ns:
        I_t, _, _ = calc_trapezoidal(f_lambda, a, b, n_)
        I_s, _, _ = calc_simpson(f_lambda, a, b, n_)
        err_trap.append(abs(I_t - I_exact))
        err_simp.append(abs(I_s - I_exact))
    
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=np.log10(ns), y=np.log10(err_trap),
                                 mode="lines+markers", name="Hình thang", line=dict(color="red")))
    fig_err.add_trace(go.Scatter(x=np.log10(ns), y=np.log10(err_simp),
                                 mode="lines+markers", name="Simpson 1/3", line=dict(color="green")))
    fig_err.update_layout(
        xaxis_title="log10(n)",
        yaxis_title="log10(Sai số)",
        height=500,
        legend_title="Công thức",
        title="Biểu đồ hội tụ sai số: độ dốc thể hiện tốc độ hội tụ"
    )
    st.plotly_chart(fig_err, use_container_width=True)

st.caption("Ứng dụng minh họa công thức Hình thang và Simpson.")
