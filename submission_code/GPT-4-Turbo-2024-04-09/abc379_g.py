import sys
input = sys.stdin.read
MOD = 998244353

def solve():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = [list(data[i + 2]) for i in range(H)]
    
    # Directions for checking adjacent cells
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Function to count valid configurations
    def count_valid_configurations():
        # Dynamic programming table
        # dp[r][c][v] represents the number of ways to fill the grid up to row r and column c
        # where the cell (r, c) is set to value v (1, 2, or 3)
        dp = [[[0 for _ in range(4)] for _ in range(W)] for _ in range(H)]
        
        # Initialize dp for the first cell
        if grid[0][0] == '?':
            for v in range(1, 4):
                dp[0][0][v] = 1
        else:
            v = int(grid[0][0])
            dp[0][0][v] = 1
        
        # Fill the dp table
        for r in range(H):
            for c in range(W):
                if r == 0 and c == 0:
                    continue
                possible_values = range(1, 4) if grid[r][c] == '?' else [int(grid[r][c])]
                for v in possible_values:
                    if (grid[r][c] != '?' and grid[r][c] != str(v)):
                        continue
                    # Sum all valid configurations from adjacent cells
                    count = 0
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            for pv in range(1, 4):
                                if pv != v:
                                    count = (count + dp[nr][nc][pv]) % MOD
                    dp[r][c][v] = count if (r != 0 or c != 0) else dp[r][c][v]
        
        # Sum up all configurations for the last cell
        result = 0
        for v in range(1, 4):
            result = (result + dp[H-1][W-1][v]) % MOD
        
        return result
    
    # Output the result
    print(count_valid_configurations())