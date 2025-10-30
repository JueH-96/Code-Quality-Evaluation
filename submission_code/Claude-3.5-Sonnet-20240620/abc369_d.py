# YOUR CODE HERE
def max_experience(N, monsters):
    dp = [0] * (N + 1)
    even_bonus = 0
    
    for i in range(N):
        # Option 1: Don't defeat this monster
        dp[i + 1] = dp[i]
        
        # Option 2: Defeat this monster
        experience = monsters[i] + even_bonus
        if i > 0:
            experience += dp[i - 1]
        
        dp[i + 1] = max(dp[i + 1], experience)
        
        # Update even_bonus for the next monster
        even_bonus = monsters[i] if even_bonus == 0 else 0
    
    return dp[N]

N = int(input())
monsters = list(map(int, input().split()))

print(max_experience(N, monsters))