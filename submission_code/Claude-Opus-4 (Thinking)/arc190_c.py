# YOUR CODE HERE
MOD = 998244353

H, W = map(int, input().split())
A = []
for _ in range(H):
    row = list(map(int, input().split()))
    A.append(row)

Q, sh, sw = map(int, input().split())
sh -= 1  # Convert to 0-indexed
sw -= 1

# Current position
r, c = sh, sw

for _ in range(Q):
    line = input().split()
    d = line[0]
    a = int(line[1])
    
    # Move
    if d == 'L':
        c -= 1
    elif d == 'R':
        c += 1
    elif d == 'U':
        r -= 1
    elif d == 'D':
        r += 1
    
    # Update grid
    A[r][c] = a
    
    # Compute dp
    dp = [[0] * W for _ in range(H)]
    dp[0][0] = A[0][0]
    
    # Fill first row
    for j in range(1, W):
        dp[0][j] = (dp[0][j-1] * A[0][j]) % MOD
    
    # Fill first column
    for i in range(1, H):
        dp[i][0] = (dp[i-1][0] * A[i][0]) % MOD
    
    # Fill rest of the table
    for i in range(1, H):
        for j in range(1, W):
            dp[i][j] = ((dp[i-1][j] + dp[i][j-1]) * A[i][j]) % MOD
    
    print(dp[H-1][W-1])