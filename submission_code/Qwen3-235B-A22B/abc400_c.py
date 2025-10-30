import math

n = int(input())
count = math.isqrt(n // 2) + math.isqrt(n // 4)
print(count)