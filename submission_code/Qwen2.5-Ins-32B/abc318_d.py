import sys
from itertools import combinations

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    idx = 1
    D = [[0] * n for _ in range(n)]
    for i in range(n-1):
        for j in range(i+1, n):
            D[i][j] = D[j][i] = int(data[idx])
            idx += 1

    def dfs(mask, last):
        if mask == (1 << n) - 1:
            return 0
        if dp[mask][last] != -1:
            return dp[mask][last]
        res = 0
        for i in range(n):
            if mask & (1 << i) == 0:
                for j in range(i+1, n):
                    if mask & (1 << j) == 0:
                        res = max(res, D[i][j] + dfs(mask | (1 << i) | (1 << j), max(i, j)))
        dp[mask][last] = res
        return res

    dp = [[-1] * n for _ in range(1 << n)]
    print(dfs(0, 0))

solve()