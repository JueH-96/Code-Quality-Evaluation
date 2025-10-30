import math

D = int(input())
min_diff = float('inf')
max_x = int(math.sqrt(D)) + 1

for x in range(0, max_x + 1):
    s = x * x
    target = D - s
    if target < 0:
        current_sum = s + 0
        diff = abs(current_sum - D)
        if diff < min_diff:
            min_diff = diff
        continue
    y_candidate = int(math.sqrt(target))
    for y in [y_candidate - 1, y_candidate, y_candidate + 1]:
        if y < 0:
            continue
        current_sum = s + y * y
        diff = abs(current_sum - D)
        if diff < min_diff:
            min_diff = diff

print(int(min_diff))