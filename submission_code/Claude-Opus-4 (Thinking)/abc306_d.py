n = int(input())

# dp[s] = max tastiness with state s
# s = 0: healthy, s = 1: upset
INF = 10**18
dp = [0, -INF]  # Initially healthy with 0 tastiness

for _ in range(n):
    x, y = map(int, input().split())
    new_dp = [-INF, -INF]
    
    # If currently healthy
    if dp[0] > -INF:
        # Skip
        new_dp[0] = max(new_dp[0], dp[0])
        
        # Eat
        if x == 0:  # Antidotal
            new_dp[0] = max(new_dp[0], dp[0] + y)
        else:  # Poisonous
            new_dp[1] = max(new_dp[1], dp[0] + y)
    
    # If currently upset
    if dp[1] > -INF:
        # Skip
        new_dp[1] = max(new_dp[1], dp[1])
        
        # Eat
        if x == 0:  # Antidotal
            new_dp[0] = max(new_dp[0], dp[1] + y)
        # If x == 1 (poisonous), we die, so we don't consider this
    
    dp = new_dp

# Answer is the maximum of healthy or upset state
ans = max(dp[0], dp[1])
print(ans)