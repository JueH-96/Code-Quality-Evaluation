import math

def find_xy_pair():
    n = int(input())
    max_cubed = 4 * n
    a = 1
    while True:
        if a ** 3 > max_cubed:
            break
        if n % a == 0:
            b = n // a
            if 4 * b >= a * a:
                D = 12 * b - 3 * a * a
                sqrtD = math.isqrt(D)
                if sqrtD * sqrtD == D:
                    numerator = -3 * a + sqrtD
                    if numerator > 0 and numerator % 6 == 0:
                        y = numerator // 6
                        x = y + a
                        print(x, y)
                        return
        a += 1
    print(-1)

find_xy_pair()