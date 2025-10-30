import bisect

n, m, p = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

b.sort()
prefix = [0] * (m + 1)
for i in range(m):
    prefix[i + 1] = prefix[i] + b[i]

total = 0
for ai in a:
    x = p - ai
    k = bisect.bisect_right(b, x)
    total += ai * k + prefix[k] + p * (m - k)

print(total)