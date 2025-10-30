import math

R = int(input())
T = R * R - 1

if T < 0:
    print(0)
else:
    # Compute y_max_x0 for x=0 case
    s_x0 = T
    sqrt_val = math.isqrt(4 * s_x0 + 1)
    y_max_x0 = (sqrt_val - 1) // 2
    x_max = y_max_x0

    term4 = 0
    for x in range(1, x_max + 1):
        s = T - (x * x + x)
        if s < 0:
            continue
        sqrt_val = math.isqrt(4 * s + 1)
        y_max = (sqrt_val - 1) // 2
        if y_max >= 1:
            term4 += y_max

    total = 1 + 4 * y_max_x0 + 4 * term4
    print(total)