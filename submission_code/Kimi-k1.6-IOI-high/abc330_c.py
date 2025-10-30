import math

D = int(input())
min_diff = D  # Initialize with the maximum possible difference
max_x = math.isqrt(D)

for x in range(max_x + 1):
    x_squared = x * x
    remaining = D - x_squared
    if remaining < 0:
        current_diff = x_squared - D
        if current_diff < min_diff:
            min_diff = current_diff
    else:
        y0 = math.isqrt(remaining)
        for y_candidate in [y0, y0 + 1]:
            sum_xy = x_squared + y_candidate * y_candidate
            current_diff = abs(sum_xy - D)
            if current_diff < min_diff:
                min_diff = current_diff

print(min_diff)