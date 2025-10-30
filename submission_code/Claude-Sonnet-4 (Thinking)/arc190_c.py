MOD = 998244353

H, W = map(int, input().split())

A = []
for _ in range(H):
    row = list(map(int, input().split()))
    A.append(row)

Q, sh, sw = map(int, input().split())

# Current position (1-indexed)
curr_h, curr_w = sh, sw

def compute_answer():
    dp = [[0] * W for _ in range(H)]
    dp[0][0] = A[0][0]
    
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                continue
            if i > 0:
                dp[i][j] = (dp[i][j] + dp[i-1][j]) % MOD
            if j > 0:
                dp[i][j] = (dp[i][j] + dp[i][j-1]) % MOD
            dp[i][j] = dp[i][j] * A[i][j] % MOD
    
    return dp[H-1][W-1]

for _ in range(Q):
    d, a = input().split()
    a = int(a)
    
    # Move Takahashi
    if d == 'L':
        curr_w -= 1
    elif d == 'R':
        curr_w += 1
    elif d == 'U':
        curr_h -= 1
    elif d == 'D':
        curr_h += 1
    
    # Update the grid (convert to 0-indexed)
    A[curr_h - 1][curr_w - 1] = a
    
    # Compute and print the answer
    print(compute_answer())