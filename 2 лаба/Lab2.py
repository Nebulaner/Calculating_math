import numpy as np

x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 
              2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9])
y = np.array([1.04, 1.47, 1.78, 2.01, 2.19, 2.60, 2.93, 3.22, 3.50, 4.01,
              4.22, 4.71, 5.23, 5.78, 6.27, 6.75, 7.16, 7.76, 8.30, 9.00])

def approximation(x, y, degree):
    A = np.vander(x, degree + 1, increasing=True)
    
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    y_fit = np.polyval(coeffs[::-1], x)
    
    mse = np.mean((y - y_fit) ** 2)
    rmse = np.sqrt(mse)
    
    return coeffs, y_fit, rmse


coeffs_1, y_fit_1, rmse_1 = approximation(x, y, degree=1)
coeffs_2, y_fit_2, rmse_2 = approximation(x, y, degree=2)

print("\nМногочлен 1-й степени (линейная аппроксимация):")
print(f"P_1(x) = {coeffs_1[0]:.6f} + {coeffs_1[1]:.6f}*x")
print(f"Среднеквадратичное отклонение: {rmse_1:.6f}")

print("\nМногочлен 2-й степени (квадратичная аппроксимация):")
print(f"P_2(x) = {coeffs_2[0]:.6f} + {coeffs_2[1]:.6f}*x + {coeffs_2[2]:.6f}*x^2")
print(f"Среднеквадратичное отклонение: {rmse_2:.6f}")