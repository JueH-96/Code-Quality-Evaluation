from math import gcd

N = int(input())
A = list(map(int, input().split()))

MOD = 998244353

def modinv(x):
    return pow(x, MOD-2, MOD)

def moddiv(x, y):
    return (x * modinv(y)) % MOD

def modsum(x, y):
    return (x + y) % MOD

def modsub(x, y):
    return (x - y) % MOD

dp = [0] * N
dp[0] = A[0]
for i in range(1, N):
    g = gcd(i+1, i)
    x = moddiv(i+1, i)
    y = moddiv(1, i)
    dp[i] = modsum(modsub(moddiv(A[i], x), moddiv(dp[i-1], y)), dp[i-1])

print(dp[-1])