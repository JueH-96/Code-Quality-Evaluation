import math

def min_diff(D):
    min_diff_val = float('inf')
    max_x = int(math.isqrt(D)) + 1
    for x in range(max_x + 1):
        x_sq = x * x
        remaining = D - x_sq
        if remaining < 0:
            y = 0
        else:
            y = int(math.isqrt(remaining))
        # Check y and y+1 to find the minimum difference
        for dy in [-1, 0, 1]:
            y_candidate = y + dy
            if y_candidate < 0:
                continue
            total = x_sq + y_candidate * y_candidate
            current_diff = abs(total - D)
            if current_diff < min_diff_val:
                min_diff_val = current_diff
            if min_diff_val == 0:
                return 0
    return min_diff_val

D = int(input())
print(min_diff(D))