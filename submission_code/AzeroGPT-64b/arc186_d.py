import sys

mod = 998244353
n = int(sys.stdin.readline().strip())
arr = list(map(int, sys.stdin.readline().strip().split()))

st, res, num_less, tot = list(range(n)), 1, 0, 0

def find(x):
    while x != st[x]:
        x = st[x]
    return x

for i in reversed(range(n)):
    res = res * pow(i + 1, arr[i], mod) * pow(i, mod - 2, mod) % mod
    st[find(i)] = find(i + 1)
    tot += 1
    if tot == n: break

dp = [1] * (n + 1)
dp[1] = 2
for i in range(2, n + 1):
    dp[i] = (dp[i - 1] * i + dp[i - 2]) % mod

for i in range(n):
    if i == st[i]:
        num_less += pow(i + 1, arr[i], mod) - 1

print(res * pow(dp[n], num_less, mod) % mod)