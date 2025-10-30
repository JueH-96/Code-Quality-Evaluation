def read_input():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(map(int, input().split())))
    Q, sh, sw = map(int, input().split())
    queries = []
    for _ in range(Q):
        d, a = input().split()
        queries.append((d, int(a)))
    return H, W, grid, Q, sh, sw, queries

def solve(H, W, grid, Q, sh, sw, queries):
    MOD = 998244353
    
    # Calculate dp arrays for each position
    def calculate_dp():
        dp_forward = [[0] * W for _ in range(H)]
        dp_backward = [[0] * W for _ in range(H)]
        
        # Initialize first position
        dp_forward[0][0] = 1
        
        # Forward dp
        for i in range(H):
            for j in range(W):
                if i > 0:
                    dp_forward[i][j] = (dp_forward[i][j] + dp_forward[i-1][j]) % MOD
                if j > 0:
                    dp_forward[i][j] = (dp_forward[i][j] + dp_forward[i][j-1]) % MOD
        
        # Initialize last position
        dp_backward[H-1][W-1] = 1
        
        # Backward dp
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                if i < H-1:
                    dp_backward[i][j] = (dp_backward[i][j] + dp_backward[i+1][j]) % MOD
                if j < W-1:
                    dp_backward[i][j] = (dp_backward[i][j] + dp_backward[i][j+1]) % MOD
                    
        return dp_forward, dp_backward

    # Calculate total paths sum
    def calculate_sum():
        total = 0
        dp_forward, dp_backward = calculate_dp()
        
        for i in range(H):
            for j in range(W):
                contribution = (dp_forward[i][j] * dp_backward[i][j]) % MOD
                contribution = (contribution * grid[i][j]) % MOD
                total = (total + contribution) % MOD
                
        return total

    # Process each query
    curr_h, curr_w = sh-1, sw-1
    for d, a in queries:
        # Move according to direction
        if d == 'U':
            curr_h -= 1
        elif d == 'D':
            curr_h += 1
        elif d == 'L':
            curr_w -= 1
        else:  # R
            curr_w += 1
            
        # Update grid value
        grid[curr_h][curr_w] = a
        
        # Calculate and print result
        print(calculate_sum())

def main():
    H, W, grid, Q, sh, sw, queries = read_input()
    solve(H, W, grid, Q, sh, sw, queries)

if __name__ == "__main__":
    main()