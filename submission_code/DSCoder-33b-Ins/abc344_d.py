# YOUR CODE HERE

import sys

def solve():
    T = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    bags = []
    for _ in range(N):
        A = list(map(str, sys.stdin.readline().split()))
        A[0] = int(A[0])
        bags.append(A)

    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0

    for i in range(N):
        for j in range(bags[i][0]):
            s = bags[i][j+1]
            for k in range(len(T) - len(s) + 1):
                if T[k:k+len(s)] == s:
                    dp[k+len(s)] = min(dp[k+len(s)], dp[k] + 1)

    if dp[-1] == float('inf'):
        print(-1)
    else:
        print(dp[-1])

solve()