# YOUR CODE HERE
import sys
import math

N = int(sys.stdin.readline())
A = []
S = []
for _ in range(N):
    line = sys.stdin.readline().strip()
    if not line:
        line = sys.stdin.readline().strip()
    a_i, s_i = line.split()
    A.append(int(a_i))
    S.append(s_i)

INF = float('inf')
dp = [[[INF]*101 for _ in range(101)] for _ in range(N+1)]

# Initial positions can be anywhere, so set dp[0][l][r] = 0 for all l and r
for l in range(1, 101):
    for r in range(1, 101):
        dp[0][l][r] = 0

for i in range(N):
    a = A[i]
    s = S[i]
    for l in range(1, 101):
        for r in range(1, 101):
            if dp[i][l][r] < INF:
                if s == 'L':
                    new_l = a
                    cost = dp[i][l][r] + abs(l - a)
                    if cost < dp[i+1][new_l][r]:
                        dp[i+1][new_l][r] = cost
                else:
                    new_r = a
                    cost = dp[i][l][r] + abs(r - a)
                    if cost < dp[i+1][l][new_r]:
                        dp[i+1][l][new_r] = cost

ans = INF
for l in range(1, 101):
    for r in range(1, 101):
        if dp[N][l][r] < ans:
            ans = dp[N][l][r]

print(ans)