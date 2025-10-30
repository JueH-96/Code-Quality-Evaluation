import bisect

m = 10**8
n = int(input())
A = list(map(int, input().split()))
a = [x % m for x in A]
a.sort()

sum_a = sum(a)
C = 0
for i in range(n):
    target = m - a[i]
    j = bisect.bisect_left(a, target, i + 1, n)
    C += n - j

sum_total = sum_a * (n - 1) - m * C
print(sum_total)