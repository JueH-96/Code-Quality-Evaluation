def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    dishes = []
    for i in range(N):
        A = int(data[3 + 2 * i])
        B = int(data[4 + 2 * i])
        dishes.append((A, B))
    
    # dp[i][j][k] means the maximum number of dishes we can eat using first i dishes
    # with total sweetness exactly j and total saltiness exactly k
    # Initialize dp array with -inf (or a very small number) to represent impossible states
    # dp[0][0][0] = 0 means using 0 dishes to achieve 0 sweetness and 0 saltiness is possible and we ate 0 dishes
    dp = [[[-float('inf')] * (Y + 1) for _ in range(X + 1)] for __ in range(N + 1)]
    dp[0][0][0] = 0
    
    for i in range(1, N + 1):
        A, B = dishes[i - 1]
        for j in range(X + 1):
            for k in range(Y + 1):
                # If we don't take the current dish
                dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k])
                # If we take the current dish
                if j >= A and k >= B:
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - A][k - B] + 1)
    
    # Find the maximum number of dishes that can be eaten without exceeding X and Y
    max_dishes = 0
    for j in range(X + 1):
        for k in range(Y + 1):
            max_dishes = max(max_dishes, dp[N][j][k])
    
    print(max_dishes)