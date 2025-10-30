# YOUR CODE HERE
import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    roulettes = []
    for _ in range(n):
        line = list(map(int, sys.stdin.readline().split()))
        roulettes.append((line[0], line[1], line[2:]))

    dp = [float('inf')] * (m + 1)
    dp[0] = 0

    for i in range(m):
        if dp[i] == float('inf'):
            continue
        for c, p, s in roulettes:
            expected_cost = c
            expected_gain = sum(s) / p
            new_dp = [0.0] * (m + 1)
            for j in range(m + 1):
                if dp[j] == float('inf'):
                    continue
                
                if j >= m:
                    new_dp[m] = min(new_dp[m] if new_dp[m] else float('inf') , dp[j])
                    continue
                
                for k in range(p):
                    nj = min(m, j + s[k])
                    new_dp[nj] += dp[j]/p if dp[j] != float('inf') else 0
            
            
            temp_dp = [float('inf')] * (m+1)
            for j in range(m+1):
                temp_dp[j] = min(dp[j], new_dp[j] + expected_cost if new_dp[j] else float('inf'))
            dp = temp_dp

    print(dp[m])

solve()