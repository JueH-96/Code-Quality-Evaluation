def main():
    import sys
    input = sys.stdin.read
    N, S, M, L = map(int, input().split())
    
    # Initialize dp array with a large number, except dp[0] = 0
    INF = float('inf')
    size = N + 13  # To cover up to N + 12
    dp = [INF] * size
    dp[0] = 0
    
    # Fill dp array
    for i in range(1, size):
        if i >= 6:
            dp[i] = min(dp[i], dp[i - 6] + S)
        if i >= 8:
            dp[i] = min(dp[i], dp[i - 8] + M)
        if i >= 12:
            dp[i] = min(dp[i], dp[i - 12] + L)
    
    # Find the minimum cost for at least N eggs
    min_cost = INF
    for i in range(N, size):
        if dp[i] < min_cost:
            min_cost = dp[i]
    
    print(min_cost)

if __name__ == "__main__":
    main()