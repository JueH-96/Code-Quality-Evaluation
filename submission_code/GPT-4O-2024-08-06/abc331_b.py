# YOUR CODE HERE
def min_cost_to_buy_eggs(N, S, M, L):
    # Initialize a large number for comparison
    INF = float('inf')
    
    # Create a dp array where dp[i] represents the minimum cost to buy at least i eggs
    dp = [INF] * (N + 1)
    
    # Base case: 0 eggs cost 0 yen
    dp[0] = 0
    
    # Iterate over each number of eggs from 0 to N
    for i in range(N + 1):
        # If we can buy a 6-egg pack
        if i + 6 <= N:
            dp[i + 6] = min(dp[i + 6], dp[i] + S)
        else:
            dp[N] = min(dp[N], dp[i] + S)
        
        # If we can buy an 8-egg pack
        if i + 8 <= N:
            dp[i + 8] = min(dp[i + 8], dp[i] + M)
        else:
            dp[N] = min(dp[N], dp[i] + M)
        
        # If we can buy a 12-egg pack
        if i + 12 <= N:
            dp[i + 12] = min(dp[i + 12], dp[i] + L)
        else:
            dp[N] = min(dp[N], dp[i] + L)
    
    # The answer is the minimum cost to buy at least N eggs
    return dp[N]

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, S, M, L = map(int, data.split())
    result = min_cost_to_buy_eggs(N, S, M, L)
    print(result)