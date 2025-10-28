import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Computation Visualizer", layout="wide")

st.sidebar.title("Select Unit")
unit = st.sidebar.radio("Units", [
    "Unit 1: Linear Dependence",
    "Unit 2: Diagonalization",
    "Unit 3: Newton–Raphson",
    "Unit 4: Interpolation",
    "Unit 5: Adams–Bashforth"
])

# ========== UNIT 1 ==========
if "Linear Dependence" in unit:
    st.title("Unit 1: Linear Dependence")
    st.write("Check whether given vectors are linearly dependent or independent.")

    n = st.number_input("Enter number of vectors:", 2, 5, 3)
    m = st.number_input("Enter dimension of each vector:", 2, 5, 3)

    vectors = []
    for i in range(n):
        vec = st.text_input(f"Vector {i+1} (comma separated):", "1,2,3")
        try:
            vectors.append([float(x) for x in vec.split(",")])
        except:
            pass

    if st.button("Check Dependence"):
        A = np.array(vectors)
        rank = np.linalg.matrix_rank(A)
        st.write(f"Matrix Rank = {rank}")
        if rank < n:
            st.error("Vectors are **Linearly Dependent**.")
        else:
            st.success("Vectors are **Linearly Independent**.")
        st.write("Augmented Matrix:")
        st.latex(sp.Matrix(A))

# ========== UNIT 2 ==========
elif "Diagonalization" in unit:
    st.title("Unit 2: Diagonalization of Linear Transformation")
    st.write("Compute eigenvalues, eigenvectors, and diagonalize a matrix.")

    mat_str = st.text_area("Enter square matrix (rows separated by semicolon):", "2,1;1,2")
    try:
        A = np.array([[float(j) for j in i.split(",")] for i in mat_str.split(";")])
        eigvals, eigvecs = np.linalg.eig(A)
        st.write("Eigenvalues:", eigvals)
        st.write("Eigenvectors:")
        st.write(eigvecs)
        if len(set(np.round(eigvals, 5))) == len(eigvals):
            st.success("Matrix is Diagonalizable.")
            D = np.diag(eigvals)
            st.write("Diagonal Matrix D:")
            st.write(D)
        else:
            st.warning("Matrix is not Diagonalizable.")
    except Exception as e:
        st.error(f"Error: {e}")

# ========== UNIT 3 ==========
elif "Newton–Raphson" in unit:
    st.title("Unit 3: Newton–Raphson Method")
    st.write("Find the root of an equation using Newton–Raphson iteration.")

    x = sp.Symbol('x')
    f_str = st.text_input("Enter function f(x):", "x**3 - x - 2")
    f = sp.sympify(f_str)
    df = sp.diff(f, x)

    x0 = st.number_input("Initial Guess:", value=1.0)
    tol = st.number_input("Tolerance:", value=0.0001)
    max_iter = st.number_input("Max Iterations:", value=20)

    if st.button("Compute Root"):
        xn = x0
        for i in range(int(max_iter)):
            fxn = float(f.subs(x, xn))
            dfxn = float(df.subs(x, xn))
            if dfxn == 0:
                st.error("Zero derivative. No solution found.")
                break
            x_next = xn - fxn/dfxn
            st.write(f"Iteration {i+1}: x = {x_next:.6f}")
            if abs(x_next - xn) < tol:
                st.success(f"Root ≈ {x_next:.6f}")
                break
            xn = x_next

# ========== UNIT 4 ==========
elif "Interpolation" in unit:
    st.title("Unit 4: Newton’s Forward and Backward Interpolation")
    st.write("Interpolate a value using given data points.")

    x_str = st.text_input("Enter x values (comma separated):", "1,2,3,4")
    y_str = st.text_input("Enter y values (comma separated):", "1,8,27,64")
    x_req = st.number_input("Enter x for interpolation:", 1.0)

    x_vals = np.array([float(i) for i in x_str.split(",")])
    y_vals = np.array([float(i) for i in y_str.split(",")])

    # Compute forward differences
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:,0] = y_vals
    for i in range(1,n):
        for j in range(n-i):
            diff_table[j][i] = diff_table[j+1][i-1] - diff_table[j][i-1]

    st.write("Forward Difference Table:")
    st.dataframe(pd.DataFrame(diff_table))

    h = x_vals[1]-x_vals[0]
    p = (x_req - x_vals[0])/h
    y_interp = y_vals[0]
    p_term = 1
    for i in range(1,n):
        p_term *= (p - (i-1))
        y_interp += (p_term * diff_table[0][i]) / np.math.factorial(i)

    st.success(f"Interpolated value at x={x_req} is {y_interp:.6f}")

# ========== UNIT 5 ==========
elif "Adams–Bashforth" in unit:
    st.title("Unit 5: Adams–Bashforth Predictor–Corrector Method")
    st.write("Solve dy/dx = f(x,y) using Adams–Bashforth 4-step method.")

    f_str = st.text_input("Enter function f(x,y):", "x + y")
    f = lambda x, y: eval(f_str)
    x0 = st.number_input("x₀:", value=0.0)
    y0 = st.number_input("y₀:", value=1.0)
    h = st.number_input("Step size (h):", value=0.1)
    xn = st.number_input("Find y at x =", value=0.4)

    N = int((xn - x0)/h)
    xs = [x0]
    ys = [y0]

    for i in range(3):  # RK4 to start
        k1 = f(xs[-1], ys[-1])
        k2 = f(xs[-1] + h/2, ys[-1] + (h*k1)/2)
        k3 = f(xs[-1] + h/2, ys[-1] + (h*k2)/2)
        k4 = f(xs[-1] + h, ys[-1] + h*k3)
        y_next = ys[-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        xs.append(xs[-1] + h)
        ys.append(y_next)

    for i in range(3, N):
        yp = ys[-1] + (h/24)*(55*f(xs[-1],ys[-1]) - 59*f(xs[-2],ys[-2]) + 37*f(xs[-3],ys[-3]) - 9*f(xs[-4],ys[-4]))
        yc = ys[-1] + (h/24)*(9*f(xs[-1]+h, yp) + 19*f(xs[-1],ys[-1]) - 5*f(xs[-2],ys[-2]) + f(xs[-3],ys[-3]))
        xs.append(xs[-1] + h)
        ys.append(yc)

    df = pd.DataFrame({"x": xs, "y": ys})
    st.dataframe(df)
    st.success(f"Approx y({xn}) = {ys[-1]:.6f}")
