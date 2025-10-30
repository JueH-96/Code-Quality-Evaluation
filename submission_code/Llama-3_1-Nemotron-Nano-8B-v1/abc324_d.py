import math

n = int(input())
s = input().strip()

sorted_s = sorted(s)
count = 0

max_x = math.isqrt(10**n - 1)
for x in range(max_x + 1):
    x_squared = x * x
    x_str = str(x_squared).zfill(n)
    if len(x_str) > n:
        continue
    if sorted(x_str) == sorted_s:
        count += 1

print(count)