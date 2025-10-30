import math

D = int(input())
min_diff = float('inf')
max_x = math.isqrt(D)

for x in range(0, max_x + 1):
    x_sq = x * x
    target = D - x_sq
    if target < 0:
        current_diff = x_sq - D
    else:
        y0 = math.isqrt(target)
        y1 = y0 + 1
        diff0 = abs(y0 * y0 - target)
        diff1 = abs(y1 * y1 - target)
        current_diff = min(diff0, diff1)
    if current_diff < min_diff:
        min_diff = current_diff
    if min_diff == 0:
        break  # Early exit if possible

print(min_diff)