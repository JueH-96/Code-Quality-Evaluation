# YOUR CODE HERE
def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        row = input().strip()
        grid.append([int(c) for c in row])
    
    min_sum = float('inf')
    
    # Try all possible column flip combinations
    for col_mask in range(1 << W):
        current_sum = 0
        
        # For each row, decide whether to flip it or not
        for i in range(H):
            # Calculate what this row looks like after column flips
            row_after_col_flips = []
            for j in range(W):
                val = grid[i][j]
                if col_mask & (1 << j):  # if column j is flipped
                    val = 1 - val
                row_after_col_flips.append(val)
            
            # Count 1s in current row state
            ones_without_row_flip = sum(row_after_col_flips)
            ones_with_row_flip = W - ones_without_row_flip
            
            # Choose the option that gives fewer 1s
            current_sum += min(ones_without_row_flip, ones_with_row_flip)
        
        min_sum = min(min_sum, current_sum)
    
    print(min_sum)

solve()