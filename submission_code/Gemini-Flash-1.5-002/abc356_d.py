# YOUR CODE HERE
MOD = 998244353

def popcount(x):
    count = 0
    while x > 0:
        count += x & 1
        x >>= 1
    return count

N, M = map(int, input().split())

ans = 0
for k in range(N + 1):
    ans = (ans + popcount(k & M)) % MOD

print(ans)