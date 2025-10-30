import math

def solve():
    N = int(input())
    max_d = int(N ** (1/3)) + 2
    found = False
    for d in range(1, max_d + 1):
        if N % d != 0:
            continue
        m = N // d
        # The equation is 3y^2 + 3d y + d^2 - m = 0
        # discriminant: 9d^2 - 12(d^2 - m) = 12m - 3d^2
        disc = 12 * m - 3 * d * d
        if disc < 0:
            continue
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc != disc:
            continue
        numerator = -3 * d + sqrt_disc
        if numerator % 6 != 0:
            continue
        y = numerator // 6
        if y > 0:
            x = y + d
            print(x, y)
            found = True
            break
    if not found:
        print(-1)

solve()