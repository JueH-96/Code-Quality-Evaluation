import math

D = int(input())
min_diff = float('inf')
max_x = int(math.sqrt(D)) + 1

for x in range(max_x + 1):
    x_sq = x * x
    if x_sq > D:
        sum_sq = x_sq
        current_diff = abs(sum_sq - D)
        if current_diff < min_diff:
            min_diff = current_diff
        continue
    else:
        target = D - x_sq
        y_floor = int(math.sqrt(target))
        for dy in (-1, 0, 1):
            y = y_floor + dy
            if y < 0:
                continue
            y_sq = y * y
            sum_sq = x_sq + y_sq
            current_diff = abs(sum_sq - D)
            if current_diff < min_diff:
                min_diff = current_diff

print(min_diff)