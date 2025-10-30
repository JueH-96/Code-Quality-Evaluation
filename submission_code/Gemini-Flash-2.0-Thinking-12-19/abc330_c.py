import math

D = int(input())
min_diff = D

x_max = int(math.sqrt(D)) + 2 # slightly larger range to be safe

for x in range(x_max + 1):
    x_squared = x * x
    if x_squared >= D:
        diff = x_squared - D
    else:
        target = D - x_squared
        y1 = int(math.sqrt(target))
        y2 = y1 + 1
        diff1 = abs(x_squared + y1 * y1 - D)
        diff2 = abs(x_squared + y2 * y2 - D)
        diff = min(diff1, diff2)
    min_diff = min(min_diff, diff)

print(min_diff)