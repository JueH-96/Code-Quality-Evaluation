from collections import defaultdict

def can_convert(s, t, k):
    if len(s) == len(t):
        return sum(1 for a, b in zip(s, t) if a != b) <= k
    
    if abs(len(s) - len(t)) > k:
        return False
    
    dp = defaultdict(lambda: float('inf'))
    dp[0, 0] = 0
    
    for i in range(len(s) + 1):
        for j in range(len(t) + 1):
            if i > 0 and j > 0:
                dp[i, j] = min(dp[i-1, j-1] + (s[i-1] != t[j-1]), dp[i, j])
            if i > 0:
                dp[i, j] = min(dp[i-1, j] + 1, dp[i, j])
            if j > 0:
                dp[i, j] = min(dp[i, j-1] + 1, dp[i, j])
    
    return dp[len(s), len(t)] <= k

k = int(input())
s = input()
t = input()

print("Yes" if can_convert(s, t, k) else "No")