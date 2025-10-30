import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))

for i in range(1, n + 1):
    idx = bisect.bisect_left(a, i)
    print(a[idx] - i)