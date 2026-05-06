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

def jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    # Диагональ и её обратные значения
    diag = np.diag(A)
    inv_diag = 1 / diag
    
    # Матрица без диагонали
    A_off = A - np.diag(diag)
    
    for k in range(max_iter):
        x_new = inv_diag * (b - np.dot(A_off, x))
        
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        
        x = x_new
    
    return x

A1 = np.array([[4, 1, 0, 1], [1, 5, -1, 0], [0, -1, 4, 1], [1, 0, 1, 5]])
b1 = np.array([6, 5, 4, 7])

A = np.array(A1)
b = np.array(b1)
    
solution = gauss(A, b)
    
if solution is not None:
    print("Решение x:", solution)
    print("Проверка Ax - b:", np.dot(A, solution) - b)
    
x = jacobi(A, b)
print("Решение:", x)
print("Невязка:", np.linalg.norm(np.dot(A, x) - b))