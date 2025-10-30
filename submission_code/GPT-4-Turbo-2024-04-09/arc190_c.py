MOD = 998244353

def read_ints():
    return map(int, input().split())

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    
    H, W = int(data[idx]), int(data[idx+1])
    idx += 2
    
    grid = []
    for _ in range(H):
        grid.append([int(x) for x in data[idx:idx+W]])
        idx += W
    
    Q, sh, sw = int(data[idx]), int(data[idx+1]), int(data[idx+2])
    idx += 3
    sh -= 1
    sw -= 1
    
    queries = []
    for _ in range(Q):
        d, a = data[idx], int(data[idx+1])
        idx += 2
        queries.append((d, a))
    
    # Precompute factorials and inverse factorials for combinations
    max_factorial = H + W - 2
    factorial = [1] * (max_factorial + 1)
    inv_factorial = [1] * (max_factorial + 1)
    
    for i in range(2, max_factorial + 1):
        factorial[i] = factorial[i-1] * i % MOD
    
    inv_factorial[max_factorial] = pow(factorial[max_factorial], MOD-2, MOD)
    for i in range(max_factorial - 1, 0, -1):
        inv_factorial[i] = inv_factorial[i + 1] * (i + 1) % MOD
    
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return factorial[n] * inv_factorial[k] % MOD * inv_factorial[n - k] % MOD
    
    # Dynamic programming to calculate the sum of products along all paths
    dp = [[0] * W for _ in range(H)]
    dp[0][0] = grid[0][0]
    
    for i in range(H):
        for j in range(W):
            if i > 0:
                dp[i][j] = (dp[i][j] + dp[i-1][j] * grid[i][j]) % MOD
            if j > 0:
                dp[i][j] = (dp[i][j] + dp[i][j-1] * grid[i][j]) % MOD
    
    # Output the initial result
    print(dp[H-1][W-1])
    
    # Process each query
    h, w = sh, sw
    for d, a in queries:
        # Move Takahashi
        if d == 'U':
            h -= 1
        elif d == 'D':
            h += 1
        elif d == 'L':
            w -= 1
        elif d == 'R':
            w += 1
        
        # Update the grid value
        grid[h][w] = a
        
        # Recalculate dp for the affected cell
        dp[h][w] = grid[h][w]
        if h > 0:
            dp[h][w] = (dp[h][w] + dp[h-1][w] * grid[h][w]) % MOD
        if w > 0:
            dp[h][w] = (dp[h][w] + dp[h][w-1] * grid[h][w]) % MOD
        
        # Output the result after this change
        print(dp[H-1][W-1])

if __name__ == "__main__":
    main()