import math

D = int(input())
min_diff = D

x_max = int(math.sqrt(D))
for x in range(x_max + 1):
    v = D - x*x
    if v >= 0:
        y0 = int(math.sqrt(v))
        diff1 = abs(x*x + y0*y0 - D)
        diff2 = abs(x*x + (y0+1)*(y0+1) - D)
        min_diff = min(min_diff, diff1, diff2)

print(min_diff)