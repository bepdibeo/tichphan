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
    expr_str = expr_str.strip()
    expr_str = expr_str.replace('^', '**')
    expr_str = expr_str.replace('ln', 'log')
    expr_str = expr_str.replace('√', 'sqrt')
    expr_str = expr_str.replace('π', 'pi')
    # không đổi 'exp' hoặc 'e' ở đây mà xử lý trong sympify locals
    return expr_str

# Đảm bảo f(x_array) trả về mảng float 1 chiều
def make_vectorized(f_lambda):
    def f_vec(x_arr):
        # x_arr: mảng chứa các điểm chia xi
        y = f_lambda(x_arr)
        y = np.asarray(y)
        # nếu là số vô hướng, mở rộng thành mảng cùng kích thước
        if y.shape == ():
            try:
                val = float(y)
            except Exception:
                # trả về array of nan để kiểm tra sau
                return np.full_like(x_arr, np.nan, dtype=float)
            return np.full_like(x_arr, val, dtype=float)
        # nếu y là mảng nhưng có kích thước không bằng kích thước của (x_arr), thử broadcast mảng
        if y.shape != x_arr.shape:
            try:
                y = np.broadcast_to(y, x_arr.shape)
            except Exception:
                # cố gắng chuyển sang mảng float 
                y = np.asarray(y, dtype=float)
                if y.shape == ():
                    return np.full_like(x_arr, float(y), dtype=float)
        # đảm bảo dữ liệu kiểu float 
        return y.astype(float)
    return f_vec

# Hai hàm công thức tính tích phân
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

# Chuẩn hóa và tạo hàm bằng SymPy/NumPy
x = sp.Symbol('x')
expr_str = normalize_expr(expr_input)

try:
    # Xử lý các trường hợp có e, pi
    f_expr = sp.sympify(expr_str, locals={'e': sp.E, 'pi': sp.pi})
    # Xử lý trường hợp hàm hằng
    is_constant = not f_expr.free_symbols  # ghi nhớ nếu là hằng
    if is_constant:
        const_val = float(f_expr)
        f_lambda = lambda t: np.full_like(t, const_val)
    else:
        f_lambda = sp.lambdify(x, f_expr, "numpy")
except Exception as e:
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
    st.error(f"Lỗi khi tính giá trị hàm trên đoạn [{a}, {b}]. Có thể hàm không xác định tại một số điểm.\n\nChi tiết: {e}")
    st.stop()

# kiểm tra số phức / vô cùng / không phải số
if np.any(np.iscomplex(Y_test)):
    st.error("Hàm trả về giá trị phức trên đoạn tích phân. Vui lòng chọn đoạn không chứa điểm làm giá trị phức.")
    st.stop()

if np.any(~np.isfinite(Y_test)):
    st.error("Hàm có giá trị vô hạn hoặc NaN trong khoảng tích phân. Vui lòng chọn đoạn không chứa tiệm cận hoặc điểm kỳ dị.")
    st.stop()

# Tính tích phân chính xác bằng SymPy (nếu được)
try:
    if is_constant:
        I_exact = float(const_val * (b - a))
    else:
        I_exact = float(sp.integrate(f_expr, (x, a, b)))
except Exception:
    I_exact = None
    st.warning("Không thể tính tích phân chính xác. Ứng dụng sẽ chỉ so sánh kết quả gần đúng nếu có thể.")
    
# Hàm tính theo sai số hoặc theo n 
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
            # giới hạn an toàn để không rơi vào vòng lặp vô tận
            if n > 10**7:
                raise RuntimeError("Không hội tụ sau nhiều bước. Hãy kiểm tra hàm hoặc tăng ε.")
    else:
        I1 = rule_func(f, a, b, n)
    return I1, n

# Hàm ước lượng sai số lý thuyết 
def theoretical_error(f_expr, a, b, n, method):
    try:
        if method == "Hình thang":
            f2 = sp.diff(f_expr, x, 2)
            f2_lamb = sp.lambdify(x, f2, "numpy")
            f2_vec = make_vectorized(f2_lamb)
            M2 = np.max(np.abs(f2_vec(np.linspace(a, b, 1000))))
            return ((b - a)**3) / (12 * n**2) * M2
        elif method == "Simpson":
            f4 = sp.diff(f_expr, x, 4)
            f4_lamb = sp.lambdify(x, f4, "numpy")
            f4_vec = make_vectorized(f4_lamb)
            M4 = np.max(np.abs(f4_vec(np.linspace(a, b, 1000))))
            return ((b - a)**5) / (180 * n**4) * M4
    except Exception:
        return None

# Tính toán chính 
I_trap = I_simp = None
n_used_trap = n_used_simp = None
err_trap = err_simp = None

# Chuẩn bị số khoảng n
if mode == "Nhập số khoảng n":
    # Chắc chắn rằng số khoảng n nhập vào là số nguyên  >=1
    n_user = int(max(1, int(n_input)))
else:
    n_user = None

# Nếu người dung chọn Simpson và số khoảng n lẻ -> cảnh báo
n_warning_msg = ""
if mode == "Nhập số khoảng n" and n_user is not None and method in ["Simpson", "Cả hai"]:
    if n_user % 2 == 1:
        n_adj = n_user + 1
        n_warning_msg = f"Chú ý: Simpson yêu cầu n chẵn —> đã tự động tăng n từ {n_user} thành {n_adj} để tính Simpson."
        st.warning(n_warning_msg)
        n_for_simp = n_adj
    else:
        n_for_simp = n_user
else:
    n_for_simp = n_user

# Tính theo phương pháp hình thang nếu được yêu cầu
if method in ["Hình thang", "Cả hai"]:
    try:
        if mode == "Nhập số khoảng n":
            I_trap, n_used_trap = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, None, n_user)
        else:
            I_trap, n_used_trap = compute_with_tolerance(f_lambda, a, b, trapezoidal_rule, epsilon, None)
        err_trap = abs(I_trap - I_exact) if I_exact is not None else None
    except Exception as e:
        st.error(f"Lỗi khi tính theo Hình thang: {e}")
        I_trap = None

# Tính theo phương pháp Simpson nếu được yêu cầu
if method in ["Simpson", "Cả hai"]:
    try:
        if mode == "Nhập số khoảng n":
            # Dùng số khoảng n đã được điều chỉnh để thỏa mãn điều kiện của Simpson
            I_simp, n_used_simp = compute_with_tolerance(f_lambda, a, b, simpson_rule, None, n_for_simp)
        else:
            I_simp, n_used_simp = compute_with_tolerance(f_lambda, a, b, simpson_rule, epsilon, None)
        err_simp = abs(I_simp - I_exact) if I_exact is not None else None
    except Exception as e:
        st.error(f"Lỗi khi tính theo Simpson: {e}")
        I_simp = None

# Hiển thị KẾT QUẢ
st.subheader("Kết quả")
cols = st.columns(3)
cols[0].metric("Tích phân chính xác", f"{I_exact:.6f}" if I_exact is not None else "—")

if method in ["Hình thang", "Cả hai"]:
    err_trap_theory = None
    if n_used_trap is not None:
        err_trap_theory = theoretical_error(f_expr, a, b, n_used_trap, "Hình thang")
    text = ""
    if err_trap is not None:
        text = f"Sai số thực nghiệm: {err_trap:.6g}"
    if err_trap_theory is not None:
        text = (text + "  |  " if text else "") + f"Sai số lý thuyết: {err_trap_theory:.6g}"
    label = f"Hình thang (n={n_used_trap})" if n_used_trap is not None else "Hình thang"
    cols[1].metric(label, f"{I_trap:.6f}" if I_trap is not None else "—", text)

if method in ["Simpson", "Cả hai"]:
    err_simp_theory = None
    if n_used_simp is not None:
        err_simp_theory = theoretical_error(f_expr, a, b, n_used_simp, "Simpson")
    text = ""
    if err_simp is not None:
        text = f"Sai số thực nghiệm: {err_simp:.6g}"
    if err_simp_theory is not None:
        text = (text + "  |  " if text else "") + f"Sai số lý thuyết: {err_simp_theory:.6g}"
    label = f"Simpson (n={n_used_simp})" if n_used_simp is not None else "Simpson"
    cols[2].metric(label, f"{I_simp:.6f}" if I_simp is not None else "—", text)

# Bảng giá trị tại các điểm chia 
st.subheader("Bảng giá trị tại các điểm chia")
def make_table(X, Y, method):
    n = len(X) - 1
    if n <= 0:
        return pd.DataFrame()
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
    total = np.sum(wf) * ( (X[-1]-X[0]) / n )  
    df = pd.DataFrame({
        "xᵢ": [f"{val:.6f}" for val in X],
        "f(xᵢ)": [f"{val:.6f}" for val in Y],
        "Trọng số (wᵢ)": [f"{val:.2f}" for val in weights],
        "wᵢ·f(xᵢ)": [f"{val:.6f}" for val in (weights * Y)]
    })
    df.loc[len(df.index)] = ["—", "—", "Tổng:", f"{np.sum(weights*Y):.6f}"]
    return df

if method in ["Hình thang", "Cả hai"] and n_used_trap is not None:
    st.markdown("**Phương pháp Hình thang**")
    X_trap = np.linspace(a, b, n_used_trap + 1)
    Y_trap = f_lambda(X_trap)
    st.dataframe(make_table(X_trap, Y_trap, "Hình thang"), use_container_width=True)

if method in ["Simpson", "Cả hai"] and n_used_simp is not None:
    st.markdown("**Phương pháp Simpson**")
    X_simp = np.linspace(a, b, n_used_simp + 1)
    Y_simp = f_lambda(X_simp)
    st.dataframe(make_table(X_simp, Y_simp, "Simpson"), use_container_width=True)

# Tùy chọn đồ thị 
st.subheader("Tùy chọn hiển thị đồ thị")
fill_toggle = st.checkbox("Hiển thị vùng tô dưới đồ thị (tích phân)", value=True)
show_parabola = st.checkbox("Hiện cung parabol nội suy cho Simpson", value=True)
interp_points = st.slider("Số điểm nội suy cho mỗi parabol (Simpson) để vẽ", min_value=10, max_value=400, value=80, step=10)

xx = np.linspace(a, b, 800)
yy = f_lambda(xx)

# Hình thang
if method in ["Hình thang", "Cả hai"] and n_used_trap is not None:
    st.subheader("Minh họa phương pháp Hình thang")
    X_trap = np.linspace(a, b, n_used_trap + 1)
    Y_trap = f_lambda(X_trap)
    fig_trap = go.Figure()
    fig_trap.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))
    if fill_toggle:
        for i in range(n_used_trap):
            xs = [X_trap[i], X_trap[i], X_trap[i+1], X_trap[i+1]]
            ys = [0.0, Y_trap[i], Y_trap[i+1], 0.0]
            fig_trap.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(255,0,0,0.25)", line=dict(color="rgba(255,0,0,0.2)"), showlegend=False))
    fig_trap.add_trace(go.Scatter(x=X_trap, y=Y_trap, mode="lines+markers", name="Các điểm chia", line=dict(color="red", dash="dot")))
    fig_trap.update_layout(xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig_trap, use_container_width=True)

# Simpson
if method in ["Simpson", "Cả hai"] and n_used_simp is not None:
    st.subheader("Minh họa phương pháp Simpson")
    X_simp = np.linspace(a, b, n_used_simp + 1)
    Y_simp = f_lambda(X_simp)
    fig_simp = go.Figure()
    fig_simp.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="f(x)", line=dict(color="blue")))

    for i in range(0, len(X_simp)-2, 2):
        xs = np.linspace(X_simp[i], X_simp[i+2], interp_points)
        coeffs = np.polyfit(
            [X_simp[i], X_simp[i+1], X_simp[i+2]],
            [Y_simp[i], Y_simp[i+1], Y_simp[i+2]], 2)
        ys = np.polyval(coeffs, xs)

        # vùng tô
        if fill_toggle:
            fig_simp.add_trace(go.Scatter(
                x=np.concatenate((xs, [xs[-1], xs[0]])),
                y=np.concatenate((ys, [0.0, 0.0])),
                fill="toself",
                fillcolor="rgba(0,200,0,0.20)",
                line=dict(color="rgba(0,200,0,0.2)"),
                showlegend=False))

        # cung parabol nội suy
        if show_parabola:
            fig_simp.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name="Cung parabol nội suy" if i==0 else None,
                line=dict(color="gold", dash="dashdot"),
                showlegend=(i==0)))

    fig_simp.add_trace(go.Scatter(
        x=X_simp, y=Y_simp,
        mode="lines+markers",
        name="Các điểm chia",
        line=dict(color="red", dash="dot")))

    fig_simp.update_layout(xaxis_title="x", yaxis_title="f(x)", height=450)
    st.plotly_chart(fig_simp, use_container_width=True)



