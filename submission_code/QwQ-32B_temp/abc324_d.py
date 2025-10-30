import math

n = int(input())
s = input().strip()

sorted_s = sorted(s)

count = 0
max_x = 10**n - 1
max_k = math.isqrt(max_x)

for k in range(max_k + 1):
    x = k * k
    s_x = str(x).zfill(n)
    if len(s_x) != n:
        continue
    if sorted(s_x) == sorted_s:
        count += 1

print(count)