# YOUR CODE HERE
N = int(input())
keys = [list(input().split()) for _ in range(N)]

def min_fatigue(keys):
    INF = float('inf')
    dp = [[INF] * 101 for _ in range(101)]
    dp[0][0] = 0

    for a, s in keys:
        a = int(a)
        new_dp = [[INF] * 101 for _ in range(101)]
        
        for l in range(101):
            for r in range(101):
                if dp[l][r] == INF:
                    continue
                
                if s == 'L':
                    new_dp[a][r] = min(new_dp[a][r], dp[l][r] + abs(a - l))
                else:
                    new_dp[l][a] = min(new_dp[l][a], dp[l][r] + abs(a - r))
        
        dp = new_dp

    return min(min(row) for row in dp)

print(min_fatigue(keys))