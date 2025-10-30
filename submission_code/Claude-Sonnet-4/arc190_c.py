MOD = 998244353

def solve():
    H, W = map(int, input().split())
    
    # Read the grid
    A = []
    for i in range(H):
        row = list(map(int, input().split()))
        A.append(row)
    
    Q, sh, sw = map(int, input().split())
    
    # Convert to 0-indexed
    sh -= 1
    sw -= 1
    
    def calculate_sum():
        # DP to calculate sum of products over all paths
        dp = [[0] * W for _ in range(H)]
        
        # Base case
        dp[0][0] = A[0][0]
        
        # Fill first row
        for j in range(1, W):
            dp[0][j] = (dp[0][j-1] * A[0][j]) % MOD
        
        # Fill first column
        for i in range(1, H):
            dp[i][0] = (dp[i-1][0] * A[i][0]) % MOD
        
        # Fill the rest
        for i in range(1, H):
            for j in range(1, W):
                dp[i][j] = (A[i][j] * (dp[i-1][j] + dp[i][j-1])) % MOD
        
        return dp[H-1][W-1]
    
    # Process queries
    for _ in range(Q):
        d, a = input().split()
        a = int(a)
        
        # Move Takahashi
        if d == 'L':
            sw -= 1
        elif d == 'R':
            sw += 1
        elif d == 'U':
            sh -= 1
        elif d == 'D':
            sh += 1
        
        # Update the grid
        A[sh][sw] = a
        
        # Calculate and print the answer
        result = calculate_sum()
        print(result)

solve()