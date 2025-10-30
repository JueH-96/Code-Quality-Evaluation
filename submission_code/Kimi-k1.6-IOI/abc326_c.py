import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
max_count = 0

for i in range(n):
    right = bisect.bisect_left(a, a[i] + m)
    max_count = max(max_count, right - i)

print(max_count)