def find_subsequences():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # dp[i][j] will store the number of ways to match the first j elements of B
    # using the first i elements of A
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    
    # Base case: There's exactly one way to match an empty B
    for i in range(N + 1):
        dp[i][0] = 1
    
    # Fill the dp table
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
            else:
                dp[i][j] = dp[i - 1][j]
    
    # Check if there are at least two subsequences matching B
    if dp[N][M] >= 2:
        print("Yes")
    else:
        print("No")