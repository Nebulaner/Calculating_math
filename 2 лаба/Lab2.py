import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка русских шрифтов
rcParams['font.family'] = 'DejaVu Sans'
# Для Windows можно использовать: rcParams['font.family'] = 'Arial'

# Исходные данные (из таблицы)
x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 
              2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9])
y = np.array([1.04, 1.47, 1.78, 2.01, 2.19, 2.60, 2.93, 3.22, 3.50, 4.01,
              4.22, 4.71, 5.23, 5.78, 6.27, 6.75, 7.16, 7.76, 8.30, 9.00])

def least_squares_approximation(x, y, degree):
    """
    Аппроксимация методом наименьших квадратов
    
    Параметры:
    x, y - массивы данных
    degree - степень многочлена
    
    Возвращает:
    coeffs - коэффициенты многочлена (от младшей степени к старшей)
    y_fit - аппроксимированные значения
    rmse - среднеквадратичное отклонение
    """
    # Строим матрицу Вандермонда
    A = np.vander(x, degree + 1, increasing=True)
    
    # Решаем систему нормальных уравнений A^T * A * c = A^T * y
    # Используем метод наименьших квадратов из numpy
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    # Вычисляем аппроксимированные значения
    y_fit = np.polyval(coeffs[::-1], x)  # polyval ожидает коэффициенты от старшей степени
    
    # Вычисляем среднеквадратичное отклонение (RMSE)
    mse = np.mean((y - y_fit) ** 2)
    rmse = np.sqrt(mse)
    
    return coeffs, y_fit, rmse

# Аппроксимация многочленами 1-й и 2-й степени
coeffs_1, y_fit_1, rmse_1 = least_squares_approximation(x, y, degree=1)
coeffs_2, y_fit_2, rmse_2 = least_squares_approximation(x, y, degree=2)

# Вывод результатов
print("=" * 60)
print("АППРОКСИМАЦИЯ МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ")
print("=" * 60)

# Вывод коэффициентов для многочлена 1-й степени
print("\nМногочлен 1-й степени (линейная аппроксимация):")
print(f"P₁(x) = {coeffs_1[0]:.6f} + {coeffs_1[1]:.6f}·x")
print(f"Среднеквадратичное отклонение (RMSE): {rmse_1:.6f}")

# Вывод коэффициентов для многочлена 2-й степени
print("\nМногочлен 2-й степени (квадратичная аппроксимация):")
print(f"P₂(x) = {coeffs_2[0]:.6f} + {coeffs_2[1]:.6f}·x + {coeffs_2[2]:.6f}·x²")
print(f"Среднеквадратичное отклонение (RMSE): {rmse_2:.6f}")

# Сравнение точности
print("\n" + "=" * 60)
print("СРАВНЕНИЕ ТОЧНОСТИ:")
print("=" * 60)
print(f"Улучшение точности при переходе к P₂: {((rmse_1 - rmse_2) / rmse_1 * 100):.2f}%")

# Таблица значений
print("\n" + "=" * 80)
print("ТАБЛИЦА ЗНАЧЕНИЙ:")
print("=" * 80)
print(f"{'x':^8} {'y (исх)':^12} {'P₁(x)':^12} {'|y-P₁|':^12} {'P₂(x)':^12} {'|y-P₂|':^12}")
print("-" * 80)
for i in range(len(x)):
    print(f"{x[i]:8.2f} {y[i]:12.4f} {y_fit_1[i]:12.4f} {abs(y[i]-y_fit_1[i]):12.4f} "
          f"{y_fit_2[i]:12.4f} {abs(y[i]-y_fit_2[i]):12.4f}")

# Построение графиков
plt.figure(figsize=(12, 8))

# График 1: Исходные данные и аппроксимации
plt.subplot(2, 1, 1)
plt.scatter(x, y, color='red', s=50, label='Исходные данные', zorder=5)

# Плавные кривые для отображения
x_smooth = np.linspace(min(x), max(x), 200)
y_smooth_1 = np.polyval(coeffs_1[::-1], x_smooth)
y_smooth_2 = np.polyval(coeffs_2[::-1], x_smooth)

plt.plot(x_smooth, y_smooth_1, 'b-', linewidth=2, 
         label=f'Линейная аппроксимация (RMSE = {rmse_1:.4f})')
plt.plot(x_smooth, y_smooth_2, 'g-', linewidth=2, 
         label=f'Квадратичная аппроксимация (RMSE = {rmse_2:.4f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Аппроксимация методом наименьших квадратов', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Остатки (погрешности)
plt.subplot(2, 1, 2)
residuals_1 = y - y_fit_1
residuals_2 = y - y_fit_2

plt.bar(x - 0.03, residuals_1, width=0.06, color='blue', alpha=0.7, 
        label=f'Остатки (линейная)')
plt.bar(x + 0.03, residuals_2, width=0.06, color='green', alpha=0.7, 
        label=f'Остатки (квадратичная)')

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('x', fontsize=12)
plt.ylabel('Погрешность', fontsize=12)
plt.title('Остатки аппроксимации', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ распределения погрешностей
print("\n" + "=" * 60)
print("АНАЛИЗ ПОГРЕШНОСТЕЙ:")
print("=" * 60)
print(f"Максимальная погрешность (линейная): {np.max(np.abs(residuals_1)):.6f}")
print(f"Максимальная погрешность (квадратичная): {np.max(np.abs(residuals_2)):.6f}")
print(f"Средняя абсолютная погрешность (линейная): {np.mean(np.abs(residuals_1)):.6f}")
print(f"Средняя абсолютная погрешность (квадратичная): {np.mean(np.abs(residuals_2)):.6f}")