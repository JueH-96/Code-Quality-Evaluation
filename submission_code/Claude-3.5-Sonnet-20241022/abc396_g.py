def solve():
    # Read input
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        row = list(map(int, list(input().strip())))
        grid.append(row)
    
    # Since W ≤ 18, we can try all possible combinations of column operations
    min_sum = float('inf')
    
    # Try all possible combinations of column operations (2^W possibilities)
    for col_mask in range(1 << W):
        # Create a copy of the grid
        new_grid = [row[:] for row in grid]
        
        # Apply column operations based on the mask
        for col in range(W):
            if col_mask & (1 << col):
                # Flip this column
                for row in range(H):
                    new_grid[row][col] = 1 - new_grid[row][col]
        
        # For each row, choose the optimal operation (flip or not)
        # After column operations, each row should be flipped if it reduces the sum
        curr_sum = 0
        for row in range(H):
            # Count ones in this row
            ones = sum(new_grid[row])
            # Take minimum of flipped and not flipped
            curr_sum += min(ones, W - ones)
        
        min_sum = min(min_sum, curr_sum)
    
    print(min_sum)

solve()