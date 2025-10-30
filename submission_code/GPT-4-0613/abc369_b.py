import sys
from heapq import heappush, heappop

def solve():
    N = int(input())
    A = [0] * (N + 2)
    S = [0] * (N + 2)
    for i in range(1, N + 1):
        A[i], S[i] = input().split()
        A[i] = int(A[i])
    A[N + 1] = 100
    S[0] = S[N + 1] = 'R'
    A[0] = 0
    dp = [[10**18] * 2 for _ in range(N + 2)]
    dp[0][0] = dp[0][1] = 0
    for i in range(N + 1):
        for j in range(2):
            if S[i] == 'L' or S[i + 1] == 'R':
                dp[i + 1][0] = min(dp[i + 1][0], dp[i][j] + abs(A[i + 1] - A[i + j]))
            if S[i] == 'R' or S[i + 1] == 'L':
                dp[i + 1][1] = min(dp[i + 1][1], dp[i][j] + abs(A[i + 1] - A[i + (j ^ 1)]))
    print(min(dp[N + 1]))

solve()