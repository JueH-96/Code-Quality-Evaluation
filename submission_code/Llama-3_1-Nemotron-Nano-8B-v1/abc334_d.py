import bisect

n, q = map(int, input().split())
r = list(map(int, input().split()))
r.sort()

prefix = [0] * (n + 1)
for i in range(1, n + 1):
    prefix[i] = prefix[i - 1] + r[i - 1]

for _ in range(q):
    x = int(input())
    k = bisect.bisect_right(prefix, x) - 1
    print(k)