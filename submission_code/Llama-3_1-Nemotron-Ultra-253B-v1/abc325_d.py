import sys

n = int(sys.stdin.readline())
products = []
for _ in range(n):
    t, d = map(int, sys.stdin.readline().split())
    products.append((t, d))

# Sort by end time (t + d), then by start time t
products.sort(key=lambda x: (x[0] + x[1], x[0]))

last = -1
count = 0

for t, d in products:
    end = t + d
    candidate = max(t, last + 1)
    if candidate <= end:
        count += 1
        last = candidate

print(count)