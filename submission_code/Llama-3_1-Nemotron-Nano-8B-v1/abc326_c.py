import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))
a.sort()

max_count = 0
for i in range(n):
    target = a[i] + m
    j = bisect.bisect_left(a, target) - 1
    current = j - i + 1
    if current > max_count:
        max_count = current

print(max_count)