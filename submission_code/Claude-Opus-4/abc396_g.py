# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    row = input().strip()
    grid.append([int(c) for c in row])

min_sum = float('inf')

# Try all possible column flip combinations
for col_mask in range(1 << W):
    current_sum = 0
    
    # For each row, determine if we should flip it
    for row in range(H):
        # Count 1s in this row after column flips
        ones_count = 0
        for col in range(W):
            # Original value
            val = grid[row][col]
            # If this column is flipped
            if col_mask & (1 << col):
                val = 1 - val
            ones_count += val
        
        # Choose whether to flip this row to minimize 1s
        current_sum += min(ones_count, W - ones_count)
    
    min_sum = min(min_sum, current_sum)

print(min_sum)