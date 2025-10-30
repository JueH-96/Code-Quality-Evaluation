import sys
from math import log2, floor

readline = sys.stdin.readline

N = int(readline())
A = list(map(int, readline().split()))
O = [[0] * (N + 1) for _ in range(30)]
exp2i = [1 << i for i in range(30)]
inf = float('inf')

for i, a in enumerate(A):
    for j in range(30):
        if exp2i[j] & a:
            O[j][i + 1] = 1
        O[j][i + 1] += O[j][i]

B = [0] * (N + 1)
for j in range(30):
    i = 0
    dp = [0, 0]
    while True:
        if i >= N:
            break
        if O[j][i] == inf:
            B[i] += 2**j * dp[0]
            i += 1
        else:
            rem = O[j][N] - O[j][i]
            p = floor(log2(rem))
            r = rem - 2**p

            B[i] += 2**j * dp[0] + 2**(j - 1) * dp[1]
            i = i + (2**p - O[j][i])

            dp[0] += 2**(j - p)
            dp[1] = r
            if p > 0:
                dp[1] += dp[0]
                dp[0] = 0

print(sum(B))