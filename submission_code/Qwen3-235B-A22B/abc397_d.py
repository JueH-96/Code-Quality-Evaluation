import math

n = int(input())

found = False

# Compute the maximum a to check
a_max = 0
while (a_max + 1) ** 3 <= 4 * n:
    a_max += 1

# Iterate over possible a values
for a in range(1, a_max + 1):
    a3 = a * a * a
    term = 4 * n - a3
    D = 3 * a * term
    
    s = math.isqrt(D)
    if s * s != D:
        continue
    
    numerator_plus = -3 * a * a + s
    if numerator_plus <= 0:
        continue
    
    if numerator_plus % (6 * a) != 0:
        continue
    
    y = numerator_plus // (6 * a)
    if y < 1:
        continue
    
    x = y + a
    print(x, y)
    found = True
    break

if not found:
    print(-1)