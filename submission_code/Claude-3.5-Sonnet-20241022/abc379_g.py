def read_input():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input().strip()))
    return H, W, grid

def is_valid(val1, val2):
    if val1 == '?' or val2 == '?':
        return True
    return val1 != val2

def solve(H, W, grid):
    MOD = 998244353
    
    # Convert grid to numeric values, with ? as 0
    numeric_grid = []
    for row in grid:
        numeric_row = []
        for cell in row:
            if cell == '?':
                numeric_row.append(0)
            else:
                numeric_row.append(int(cell))
        numeric_grid.append(numeric_row)
    
    # Calculate total states needed for DP
    total_states = 1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                total_states *= 3
                if total_states >= MOD:
                    total_states %= MOD
    
    # Initialize DP array
    dp = {tuple(): 1}  # Empty state has 1 way
    
    # Process each cell
    for i in range(H):
        for j in range(W):
            new_dp = {}
            
            # For each previous state
            for state, count in dp.items():
                # Try each possible value (1, 2, 3)
                for val in range(1, 4):
                    # If current cell is not '?' and doesn't match val, skip
                    if numeric_grid[i][j] != 0 and numeric_grid[i][j] != val:
                        continue
                        
                    # Check if val conflicts with adjacent cells
                    valid = True
                    
                    # Check left cell
                    if j > 0:
                        left_val = state[-1] if state else numeric_grid[i][j-1]
                        if left_val == val:
                            valid = False
                    
                    # Check upper cell
                    if i > 0:
                        up_val = numeric_grid[i-1][j]
                        if up_val == val:
                            valid = False
                    
                    if valid:
                        new_state = state + (val,)
                        new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
            
            dp = new_dp
    
    # Sum up all valid final states
    result = sum(dp.values()) % MOD
    return result

def main():
    H, W, grid = read_input()
    result = solve(H, W, grid)
    print(result)

if __name__ == "__main__":
    main()