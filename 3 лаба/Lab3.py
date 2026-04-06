import numpy as np

def spline(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1
    
    h = np.diff(x)
    
    a = y[:-1].copy()
    
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    
    if n >= 2:
        A[0, 0] = -h[1]
        A[0, 1] = h[0] + h[1]
        A[0, 2] = -h[0]
        rhs[0] = 0
        
        A[n, n-2] = -h[n-1]
        A[n, n-1] = h[n-2] + h[n-1]
        A[n, n] = -h[n-2]
        rhs[n] = 0
    else:
        A[0, 0] = 1
        rhs[0] = 0
        A[n, n] = 1
        rhs[n] = 0
    
    c = np.linalg.solve(A, rhs)
    
    b = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    return a, b, c, d

x = [1.000, 1.510, 2.100, 2.750, 3.420, 3.915, 4.350, 4.800, 5.200]
y = [0.0000, 0.2729, 0.3533, 0.3679, 0.3595, 0.3486, 0.3380, 0.3268, 0.3171]

a, b, c, d = spline(x, y)

a = np.round(a, 4)
b = np.round(b, 4)
c = np.round(c, 4)
d = np.round(d, 4)

for i in range(len(a)):
        print(f"\nИнтервал {i+1}: [{x[i]:.4f}, {x[i+1]:.4f}]")
        print(f"  S_{i}(x) = a_{i} + b_{i}*(x-{x[i]:.4f}) + c_{i}*(x-{x[i]:.4f})^2 + d_{i}*(x-{x[i]:.4f})^3")
        print(f"  a_{i} = {a[i]:.4f}")
        print(f"  b_{i} = {b[i]:.4f}")
        print(f"  c_{i} = {c[i]:.4f}")
        print(f"  d_{i} = {d[i]:.4f}")