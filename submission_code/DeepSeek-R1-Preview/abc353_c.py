import bisect

n = int(input())
A = list(map(int, input().split()))
MOD = 10**8

S = sum(A)
A_sorted = sorted(A)

C = 0
for i in range(n-1):
    target = MOD - A_sorted[i]
    j = bisect.bisect_left(A_sorted, target, i + 1, n)
    C += (n - j)

ans = S * (n - 1) - MOD * C
print(ans)