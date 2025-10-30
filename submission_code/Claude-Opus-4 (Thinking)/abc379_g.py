# YOUR CODE HERE
def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    MOD = 998244353
    
    # Transpose if necessary to minimize the DP state size
    if W > H:
        # Transpose
        new_grid = []
        for j in range(W):
            row = ""
            for i in range(H):
                row += grid[i][j]
            new_grid.append(row)
        grid = new_grid
        H, W = W, H
    
    # Generate all valid colorings for a row
    def generate_row_colorings(row_str):
        colorings = []
        
        def backtrack(col, current):
            if col == W:
                colorings.append(tuple(current))
                return
            
            if row_str[col] == '?':
                possible = [1, 2, 3]
            else:
                possible = [int(row_str[col])]
            
            for val in possible:
                # Check horizontal constraint
                if col > 0 and current[-1] == val:
                    continue
                
                current.append(val)
                backtrack(col + 1, current)
                current.pop()
        
        backtrack(0, [])
        return colorings
    
    # Special case for single row
    if H == 1:
        colorings = generate_row_colorings(grid[0])
        print(len(colorings))
        return
    
    # dp[coloring] = number of ways to reach this coloring for the current row
    dp = {}
    
    # Initialize first row
    for coloring in generate_row_colorings(grid[0]):
        dp[coloring] = 1
    
    # Process subsequent rows
    for row in range(1, H):
        new_dp = {}
        row_colorings = generate_row_colorings(grid[row])
        
        for new_coloring in row_colorings:
            count = 0
            for prev_coloring, prev_count in dp.items():
                # Check if compatible (no same colors in same column)
                compatible = True
                for col in range(W):
                    if prev_coloring[col] == new_coloring[col]:
                        compatible = False
                        break
                
                if compatible:
                    count = (count + prev_count) % MOD
            
            if count > 0:
                new_dp[new_coloring] = count
        
        dp = new_dp
        
        # Early termination if no valid colorings
        if not dp:
            print(0)
            return
    
    # Sum all valid colorings
    result = sum(dp.values()) % MOD
    print(result)

solve()