def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    S = int(data[1])
    M = int(data[2])
    L = int(data[3])
    
    # We will use dynamic programming to solve this problem.
    # dp[i] will store the minimum cost to get at least i eggs.
    dp = [float('inf')] * (N + 1)
    dp[0] = 0  # Base case: 0 cost for 0 eggs
    
    # We need to consider each possible pack and update the dp array accordingly.
    for i in range(N + 1):
        if i + 6 <= N:
            dp[i + 6] = min(dp[i + 6], dp[i] + S)
        else:
            dp[N] = min(dp[N], dp[i] + S)
        
        if i + 8 <= N:
            dp[i + 8] = min(dp[i + 8], dp[i] + M)
        else:
            dp[N] = min(dp[N], dp[i] + M)
        
        if i + 12 <= N:
            dp[i + 12] = min(dp[i + 12], dp[i] + L)
        else:
            dp[N] = min(dp[N], dp[i] + L)
    
    # We need to ensure that we have the minimum cost for at least N eggs.
    # Since we might have updated dp[N] from positions less than N, we should take the minimum from N onwards.
    min_cost = dp[N]
    
    print(min_cost)

if __name__ == "__main__":
    main()