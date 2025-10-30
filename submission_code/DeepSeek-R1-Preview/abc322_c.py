import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))

for i in range(1, n + 1):
    j = bisect.bisect_left(a, i)
    print(a[j] - i)