import math

def lagrange(points, x):
    n = len(points)
    res = 0.0
    
    for i in range(n):
        xi, yi = points[i]
        t = 1.0
        for j in range(n):
            if j != i:
                xj, _ = points[j]
                t *= (x - xj) / (xi - xj)
        res += yi * t
    
    return res

def aitken(points, x):
    n = len(points)
    
    p = [[0] * n for _ in range(n)]
    for i in range(n):
        p[i][i] = points[i][1]
    
    for k in range(1, n):
        for i in range(n - k):
            j = i + k
            xi, yi = points[i]
            xj, yj = points[j]
            p[i][j] = (p[i][j-1] * (xj - x) - p[i+1][j] * (xi - x)) / (xj - xi)
    
    return p[0][n-1]

def newton1(points, x):
    n = len(points)
    
    points = sorted(points, key=lambda p: p[0])
    
    work = [points[i][1] for i in range(n)]
    
    res = work[0]
    prod = 1.0
    
    for j in range(1, n):
        for i in range(n - j):
            work[i] = (work[i+1] - work[i]) / (points[i+j][0] - points[i][0])
        
        prod *= (x - points[j-1][0])
        res += work[0] * prod
    
    return res

def newton2(x, h, a, b):
    f = lambda x: math.exp(1) - (math.log(x)) ** 2
    
    n = int((b - a) / h) + 1
    points = [(a + i*h, f(a + i*h)) for i in range(n)]
    
    y = [p[1] for p in points]
    diff = [y.copy()]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            row.append(diff[j-1][i+1] - diff[j-1][i])
        diff.append(row)
    
    if x > (a + b) / 2:
        t = (x - points[-1][0]) / h
        res = diff[0][-1]
        term = 1
        fact = 1
        for j in range(1, n):
            term *= (t + j - 1)
            fact *= j
            res += diff[j][-1] * term / fact
    else:
        t = (x - points[0][0]) / h
        res = diff[0][0]
        term = 1
        fact = 1
        for j in range(1, n):
            term *= (t - j + 1)
            fact *= j
            res += diff[j][0] * term / fact
    
    return res

def stirling(x, h, a, b):
    f = lambda x: math.exp(1) - (math.log(x)) ** 2
    
    n = int((b - a) / h) + 1
    points = [(a + i*h, f(a + i*h)) for i in range(n)]
    
    y = [p[1] for p in points]
    diff = [y.copy()]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            row.append(diff[j-1][i+1] - diff[j-1][i])
        diff.append(row)
    
    center = n // 2
    x0 = points[center][0]
    t = (x - x0) / h
    
    res = diff[0][center]
    
    if n > 1:
        res += t * (diff[1][center-1] + diff[1][center]) / 2
    
    if n > 2:
        res += t**2 / 2 * diff[2][center-1]
    
    if n > 3:
        res += t * (t**2 - 1) / 6 * (diff[3][center-2] + diff[3][center-1]) / 2
    
    if n > 4:
        res += t**2 * (t**2 - 1) / 24 * diff[4][center-2]
    
    return res

points = [
    (11.153, -3.234),
    (11.454, 5.321),
    (11.673, -1.123),
    (11.879, 0.393),
    (12.009, 8.939),
    (12.231, 141.231),
    (12.549, 15.001)
]
h = 1
a = 1
b = 5

x = 12.776
print("Результат по Лагранжу в точке " + str(x) + ": " + str(round(lagrange(points, x), 2)))
print("Результат по Эйткену в точке " + str(x) + ": " + str(round(aitken(points, x), 2)))
print("")

x1 = 11.515
x2 = 11.995
print("Результат по Ньютону с разделёнными разностями в точке " + str(x1) + ": " + str(round(newton1(points, x1), 2)))
print("Результат по Ньютону с  разделёнными разностями в точке " + str(x2) + ": " + str(round(newton1(points, x2), 2)))
print("")


x1 = 0.77
x2 = 4.82
x = 3.2
print("Результат по Ньютону с конечными разностями в точке " + str(x1) + ": " + str(round(newton2(x1, h, a, b), 2)))
print("Результат по Ньютону с конечными разностями в точке " + str(x2) + ": " + str(round(newton2(x2, h, a, b), 2)))
print("Результат по Стирлингу в точке " + str(x) + ": " + str(round(stirling(x, h, a, b), 2)))
print("")