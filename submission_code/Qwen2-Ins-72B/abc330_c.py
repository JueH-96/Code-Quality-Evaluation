import math
D = int(input())

sqrtD = int(math.sqrt(D))
min_diff = D

for x in range(sqrtD + 1):
    for y in range(sqrtD + 1):
        diff = abs(x**2 + y**2 - D)
        if diff < min_diff:
            min_diff = diff

print(min_diff)