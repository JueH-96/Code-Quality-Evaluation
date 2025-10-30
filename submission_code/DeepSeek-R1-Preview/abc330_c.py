import math

D = int(input())
min_diff = float('inf')
x_max = int(math.isqrt(D)) + 1

for x in range(x_max + 1):
    x_sq = x * x
    rem = D - x_sq
    if rem < 0:
        diff = x_sq - D
        if diff < min_diff:
            min_diff = diff
        continue
    y = math.isqrt(rem)
    candidates = [y - 1, y, y + 1]
    for y_cand in candidates:
        if y_cand < 0:
            continue
        y_sq = y_cand ** 2
        total = x_sq + y_sq
        diff = abs(total - D)
        if diff < min_diff:
            min_diff = diff
            if min_diff == 0:
                print(0)
                exit()

print(min_diff)