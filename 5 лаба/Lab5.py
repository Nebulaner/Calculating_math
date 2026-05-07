import numpy as np

def gauss(A, b):
    n = A.shape[0]
    
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if np.abs(Ab[max_row, i]) < 1e-12:
            continue
        
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            print(f"Переставлены строки {i+1} и {max_row+1}")
        
        Ab[i] = Ab[i] / Ab[i, i]
        
        for j in range(n):
            if j != i:
                Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
    
    rank_A = np.linalg.matrix_rank(A, tol=1e-10)
    rank_Ab = np.linalg.matrix_rank(Ab, tol=1e-10)
    
    if rank_A == rank_Ab:
        if rank_A == n:
            print("Система имеет единственное решение.")
            return Ab[:, -1]
        else:
            print("Система имеет бесконечно много решений.")
            print("Частное решение:")
            return Ab[:, -1]
    else:
        print("Система несовместна.")
        return None

def jacobi(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    x = np.zeros(n)
    
    diag = np.diag(A)
    inv_diag = 1 / diag
    
    A_off = A - np.diag(diag)
    
    for k in range(1000):
        x_new = inv_diag * (b - np.dot(A_off, x))
        
        if np.linalg.norm(x_new - x, np.inf) < 1e-6:
            return x_new
        
        x = x_new
    
    return x

def seidel(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    x = np.zeros(n)

    for k in range(1000):
        x_old = x.copy()
        
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        
        if np.linalg.norm(x - x_old, np.inf) < 1e-6:
            return x
    
    return x


A = np.array([[0.97, 0.05, -0.22, 0.33], [-0.22, 0.45, 0.08, -0.07], [-0.33, -0.13, 1.08, 0.05], [-0.08, -0.17, -0.29, 0.67]])
b = np.array([0.43, -1.8, -0.8, 1.7])

print("ГАУСС")
x = gauss(A, b)
if x is not None:
    print("Решение x:", x)
    
print()

print("ЯКОБИ")
x = jacobi(A, b)
print("Решение:", x)

print()

print("ЗЕЙДЕЛЬ")
x = seidel(A, b)
print("Решение:", x)