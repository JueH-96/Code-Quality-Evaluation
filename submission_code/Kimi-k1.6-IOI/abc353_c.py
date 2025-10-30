import bisect

n = int(input())
a = list(map(int, input().split()))
a.sort()
mod = 10**8
sum_a = sum(a)
sum_total = sum_a * (n - 1)
count = 0

for i in range(n):
    target = mod - a[i]
    j = bisect.bisect_left(a, target, i + 1, n)
    count += (n - j)

ans = sum_total - count * mod
print(ans)