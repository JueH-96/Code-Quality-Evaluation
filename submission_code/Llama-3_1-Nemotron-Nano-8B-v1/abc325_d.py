import sys

n = int(sys.stdin.readline())
products = []

for _ in range(n):
    t, d = map(int, sys.stdin.readline().split())
    end = t + d
    products.append((end, t))

products.sort()

last_time = -1
count = 0

for end, t in products:
    earliest = max(t, last_time + 1)
    if earliest <= end:
        count += 1
        last_time = earliest

print(count)