import numpy as np

def gauss(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
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
    
A = [
    [2, 1, -1, 3],
    [4, 2, -1, 7],
    [1, 1, 1, 1],
    [3, 1, -2, 5]
    ]
    
b = [1, 3, 2, 0]
    
solution = gauss(A, b)
    
if solution is not None:
    print("Решение x:", solution)
    print("Проверка Ax - b:", np.dot(A, solution) - b)