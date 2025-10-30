# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    C = data[1:N+1]
    
    # Initialize the DP table
    dp = [[[-1] * (2 * N + 1) for _ in range(N)] for _ in range(N)]
    
    # Base case for k = 0
    for i in range(N):
        dp[i][i][0] = 0
    
    # Base case for k = 1
    for i in range(N):
        for j in range(N):
            if C[i][j] != '-':
                if i == j or C[i][j] == C[j][i]:
                    dp[i][j][1] = 1
    
    # Fill the DP table
    for k in range(2, 2 * N + 1):
        for i in range(N):
            for j in range(N):
                for u in range(N):
                    for v in range(N):
                        if C[i][u] != '-' and C[v][j] != '-' and C[i][u] == C[v][j]:
                            if dp[u][v][k-2] != -1:
                                if dp[i][j][k] == -1:
                                    dp[i][j][k] = dp[u][v][k-2] + 2
                                else:
                                    dp[i][j][k] = min(dp[i][j][k], dp[u][v][k-2] + 2)
    
    # Output the result
    result = []
    for i in range(N):
        row = []
        for j in range(N):
            min_length = -1
            for k in range(2 * N + 1):
                if dp[i][j][k] != -1:
                    if min_length == -1 or dp[i][j][k] < min_length:
                        min_length = dp[i][j][k]
            row.append(min_length)
        result.append(' '.join(map(str, row)))
    
    print('
'.join(result))