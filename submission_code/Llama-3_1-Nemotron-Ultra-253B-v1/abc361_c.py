import bisect

n, k = map(int, input().split())
a = list(map(int, input().split()))
a.sort()
m = n - k
min_diff = float('inf')

for i in range(len(a) - m + 1):
    low = a[i]
    high = a[i + m - 1]
    left = bisect.bisect_left(a, low)
    right = bisect.bisect_right(a, high)
    if right - left >= m:
        min_diff = min(min_diff, high - low)

print(min_diff)