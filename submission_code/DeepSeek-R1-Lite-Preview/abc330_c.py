import math

D = int(input())

overall_min_diff = float('inf')
x_max = math.isqrt(D)

for x in range(x_max + 1):
    target = D - x * x
    if target < 0:
        diff = abs(x * x - D)
        if diff < overall_min_diff:
            overall_min_diff = diff
    else:
        y_floor = math.isqrt(target)
        min_diff_x = float('inf')
        # Check y_floor -1
        if y_floor - 1 >= 0:
            diff = abs(x * x + (y_floor - 1) * (y_floor - 1) - D)
            if diff < min_diff_x:
                min_diff_x = diff
        # Check y_floor
        diff = abs(x * x + y_floor * y_floor - D)
        if diff < min_diff_x:
            min_diff_x = diff
        # Check y_floor +1
        diff = abs(x * x + (y_floor + 1) * (y_floor + 1) - D)
        if diff < min_diff_x:
            min_diff_x = diff
        # Check y_floor +2
        y = y_floor + 2
        if y * y <= D + x * x:
            diff = abs(x * x + y * y - D)
            if diff < min_diff_x:
                min_diff_x = diff
        # Update overall_min_diff
        if min_diff_x < overall_min_diff:
            overall_min_diff = min_diff_x

print(overall_min_diff)