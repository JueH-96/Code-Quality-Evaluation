import sys

N = int(input())
A = list(map(int, input().split()))

mod = 998244353

pow10 = [1]
for _ in range(N):
    pow10.append(pow10[-1] * 10 % mod)

A_pow10 = [a * pow10[i] for i, a in enumerate(A)]

s1 = sum(A) % mod
s2 = sum(A_pow10) % mod

ans = 0
for i, a in enumerate(A):
    s1 -= a
    s2 -= A_pow10[i]
    ans += a * s1 + A_pow10[i] * (N - 1 - i) - A_pow10[i] * i
    ans %= mod

print(ans)