# YOUR CODE HERE
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve():
    n = int(input())
    checkpoints = []
    for _ in range(n):
        x, y = map(int, input().split())
        checkpoints.append((x, y))
    
    # dp[i] = minimum cost to reach checkpoint i
    dp = [float('inf')] * n
    dp[0] = 0  # Starting at checkpoint 1 (index 0)
    
    # For each checkpoint i
    for i in range(1, n):
        # Try coming from each previous checkpoint j
        for j in range(i):
            # Calculate distance from j to i
            dist = distance(checkpoints[j], checkpoints[i])
            
            # Calculate number of checkpoints skipped
            skipped = i - j - 1
            
            # Calculate penalty
            if skipped > 0:
                penalty = 2 ** (skipped - 1)
            else:
                penalty = 0
            
            # Update dp[i] with the minimum cost
            dp[i] = min(dp[i], dp[j] + dist + penalty)
    
    print(dp[n-1])

solve()