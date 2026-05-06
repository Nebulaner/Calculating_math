import numpy as np

def f(x):
    return (2*x + 3)**6

def newton_cotes(a, b):
    x_vals = np.linspace(a, b, 6)
    h = (b - a) / 5

    weights_raw = np.array([19, 75, 50, 50, 75, 19])
    weights = weights_raw * h / 288
    
    return np.sum(weights * f(x_vals))

def gauss(a, b):
    nodes_1 = np.array([-0.8611363115940526, -0.3399810435848563,
                        0.3399810435848563, 0.8611363115940526])
    weights_1 = np.array([0.3478548451374538, 0.6521451548625461,
                          0.6521451548625461, 0.3478548451374538])
    
    x = 0.5*(b - a)*nodes_1 + 0.5*(a + b)
    w = 0.5*(b - a)*weights_1
    
    return np.sum(w * f(x))

print("Start")
a, b = 1, 8
I_exact = (19**7 - 5**7) / 14
I_nc = newton_cotes(a, b)
I_gauss = gauss(a, b)

error_nc = abs(I_exact - I_nc)
error_gauss = abs(I_exact - I_gauss)

print("Результаты:")
print(f"Точное значение: {I_exact:.2f}")
print(f"Ньютон-Котес (n=5): {I_nc:.2f}, ошибка: {error_nc:.2e}")
print(f"Гаусс (n=4): {I_gauss:.2f}, ошибка: {error_gauss:.2e}")