# YOUR CODE HERE
def count_remaining_sets(N, X, Y):
    MOD = 998244353
    
    # Create a list of balls with their coordinates
    balls = list(zip(X, Y))
    
    # Sort balls based on X coordinate
    balls.sort()
    
    # Initialize dp array
    dp = [1] * (N + 1)
    
    # Iterate through balls
    for i in range(N):
        for j in range(i):
            if balls[j][1] < balls[i][1]:
                dp[i] = (dp[i] + dp[j]) % MOD
    
    # Sum up all possible sets
    return sum(dp) % MOD

# Read input
N = int(input())
X, Y = [], []
for _ in range(N):
    x, y = map(int, input().split())
    X.append(x)
    Y.append(y)

# Calculate and print the result
result = count_remaining_sets(N, X, Y)
print(result)