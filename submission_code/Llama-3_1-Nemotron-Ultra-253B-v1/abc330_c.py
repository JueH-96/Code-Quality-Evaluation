import math

D = int(input())
x_max = math.isqrt(D)
min_diff = D

for x in range(x_max + 1):
    x_sq = x * x
    remaining = D - x_sq
    if remaining < 0:
        current_diff = x_sq - D
        if current_diff < min_diff:
            min_diff = current_diff
        continue
    y_floor = math.isqrt(remaining)
    for dy in (-1, 0, 1):
        y = y_floor + dy
        if y < 0:
            continue
        y_sq = y * y
        current_sum = x_sq + y_sq
        current_diff = abs(current_sum - D)
        if current_diff < min_diff:
            min_diff = current_diff

x_plus = x_max + 1
current_diff = x_plus * x_plus - D
if current_diff < min_diff:
    min_diff = current_diff

print(min_diff)