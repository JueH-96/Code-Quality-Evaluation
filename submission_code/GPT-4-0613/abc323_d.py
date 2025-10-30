import sys
from heapq import *
from collections import deque

def solve():
    N = int(input())
    SC = sorted([list(map(int, input().split())) for _ in range(N)])
    SC.append([10**18, 0])
    dp = [0]*(N+1)
    dp[0] = SC[0][1]
    que = deque([(SC[0][0], SC[0][1])])
    for i in range(1, N+1):
        S, C = SC[i]
        while que and que[-1][1] < C:
            _, C2 = que.pop()
            dp[i] = max(dp[i], dp[i-1]+C2)
        if que:
            dp[i] = max(dp[i], dp[i-1]+C-que[-1][1])
        else:
            dp[i] = max(dp[i], dp[i-1]+C)
        que.append((S, C))
    print(sum(SC[i][1] for i in range(N+1))-dp[-1])

solve()