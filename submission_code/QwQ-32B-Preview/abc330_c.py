import math
import bisect

D = int(input())

max_xy = int(math.isqrt(D))
y_squared = [y * y for y in range(max_xy + 1)]

min_diff = float('inf')

for x in range(max_xy + 1):
    x_squared = x * x
    target = D - x_squared
    idx = bisect.bisect_left(y_squared, target)
    
    # Check the y_squared at index idx
    if idx < len(y_squared):
        diff = abs(x_squared + y_squared[idx] - D)
        if diff < min_diff:
            min_diff = diff
    # Check the y_squared at index idx - 1
    if idx - 1 >= 0:
        diff = abs(x_squared + y_squared[idx - 1] - D)
        if diff < min_diff:
            min_diff = diff

print(min_diff)