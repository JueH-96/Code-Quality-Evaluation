import math

n = int(input())
s = input().strip()

sorted_s = sorted(s)
count = 0

max_root = math.isqrt(10**n - 1)

for i in range(0, max_root + 1):
    square = i * i
    padded = str(square).zfill(n)
    if sorted(padded) == sorted_s:
        count += 1

print(count)