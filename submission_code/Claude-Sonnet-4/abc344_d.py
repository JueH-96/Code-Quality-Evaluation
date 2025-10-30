# YOUR CODE HERE
def solve():
    T = input().strip()
    N = int(input().strip())
    
    bags = []
    for _ in range(N):
        line = input().strip().split()
        A = int(line[0])
        strings = line[1:A+1]
        bags.append(strings)
    
    # dp[i][j] = minimum cost to match first i characters of T using first j bags
    # Initialize with infinity
    INF = float('inf')
    dp = [[INF] * (N + 1) for _ in range(len(T) + 1)]
    
    # Base case: empty string can be matched with 0 cost using any number of bags
    for j in range(N + 1):
        dp[0][j] = 0
    
    # Fill DP table
    for j in range(1, N + 1):  # for each bag
        for i in range(len(T) + 1):  # for each position in T
            # Option 1: don't use bag j
            dp[i][j] = dp[i][j-1]
            
            # Option 2: use a string from bag j
            for s in bags[j-1]:
                # Check if string s can be used at position i
                if i + len(s) <= len(T) and T[i:i+len(s)] == s:
                    if dp[i][j-1] != INF:
                        dp[i + len(s)][j] = min(dp[i + len(s)][j], dp[i][j-1] + 1)
    
    result = dp[len(T)][N]
    if result == INF:
        print(-1)
    else:
        print(result)

solve()