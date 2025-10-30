import sys
from collections import defaultdict

def solve():
    n = int(sys.stdin.readline().strip())
    a = list(map(int, sys.stdin.readline().strip().split()))

    graph = defaultdict(list)
    for i in range(n):
        graph[a[i]].append(i+1)

    dp = [0] * (n+1)
    dp[0] = 1
    for i in range(1, n+1):
        dp[i] = dp[i-1] * 2
        if i in graph:
            for j in graph[i]:
                dp[j] = dp[i-1] + 1

    print(sum(dp) - n)

solve()