import sys
import math

D = int(sys.stdin.readline())

ans = D
x = 0
while True:
    x2 = x * x
    if x2 > D + ans:
        break
    t = D - x2
    if t >= 0:
        y0 = math.isqrt(t)
        for y in [y0 - 1, y0, y0 + 1]:
            if y >= 0:
                delta = abs(x2 + y * y - D)
                if delta < ans:
                    ans = delta
    else:
        delta = x2 - D
        if delta < ans:
            ans = delta
    x += 1

print(ans)