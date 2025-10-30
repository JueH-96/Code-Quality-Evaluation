import math

def min_diff(D):
    min_diff_val = float('inf')
    max_x = int(math.isqrt(D)) + 1
    for x in range(max_x + 1):
        x_squared = x * x
        if x_squared > D:
            break
        remaining = D - x_squared
        y = int(math.isqrt(remaining))
        for dy in [-1, 0, 1]:
            y_candidate = y + dy
            if y_candidate < 0:
                continue
            y_squared = y_candidate * y_candidate
            current_diff = abs(x_squared + y_squared - D)
            if current_diff < min_diff_val:
                min_diff_val = current_diff
            if min_diff_val == 0:
                return 0
    return min_diff_val

D = int(input())
print(min_diff(D))