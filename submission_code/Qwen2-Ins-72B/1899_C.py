from sys import stdin, stdout
from itertools import accumulate

def solve():
    for _ in range(int(stdin.readline())):
        n = int(stdin.readline())
        a = list(map(int, stdin.readline().split()))
        dp = [[0, 0] for _ in range(n)]
        dp[0][a[0] % 2] = a[0]
        for i in range(1, n):
            dp[i][a[i] % 2] = max(dp[i - 1][(a[i] + 1) % 2], 0) + a[i]
        stdout.write(f"{max(map(max, dp))}
")

solve()