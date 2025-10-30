def min_grid_sum(H, W, grid):
    min_sum = float('inf')
    
    # Iterate over all possible column flip combinations
    for col_mask in range(1 << W):
        # Calculate the effect of column flips
        flipped_grid = []
        for row in grid:
            new_row = []
            for j in range(W):
                if (col_mask & (1 << j)) != 0:
                    # Flip the column
                    new_row.append(1 - row[j])
                else:
                    new_row.append(row[j])
            flipped_grid.append(new_row)
        
        # Calculate the minimum sum for this column flip configuration
        total_ones = 0
        for row in flipped_grid:
            row_ones = sum(row)
            # Decide whether to flip the row or not
            total_ones += min(row_ones, W - row_ones)
        
        # Update the minimum sum
        min_sum = min(min_sum, total_ones)
    
    return min_sum

# Reading input
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = []

index = 2
for _ in range(H):
    grid.append([int(c) for c in data[index]])
    index += 1

# Calculate and print the result
result = min_grid_sum(H, W, grid)
print(result)