import math

D = int(input())
min_diff = D
max_x = math.isqrt(D)

for x in range(max_x + 1):
    s = D - x * x
    if s < 0:
        continue
    y0 = math.isqrt(s)
    diff1 = s - y0 * y0
    diff2 = (y0 + 1) ** 2 - s
    current_min = min(diff1, diff2)
    if current_min < min_diff:
        min_diff = current_min
        if min_diff == 0:
            print(0)
            exit()

print(min_diff)