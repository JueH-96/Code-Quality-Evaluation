import math

D = int(input())

# Find the minimum value of |x^2 + y^2 - D|
min_diff = float('inf')
for x in range(int(math.sqrt(D)) + 1):
    for y in range(int(math.sqrt(D)) + 1):
        diff = abs(x**2 + y**2 - D)
        min_diff = min(min_diff, diff)

print(min_diff)