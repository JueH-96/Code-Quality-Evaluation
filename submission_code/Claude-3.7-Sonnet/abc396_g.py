def min_sum_after_operations(H, W, grid_strings):
    min_sum = float('inf')
    
    # Convert grid_strings to integers
    grid_ints = [int(row, 2) for row in grid_strings]
    
    # Brute force all possible column flip combinations
    for mask in range(1 << W):
        total_sum = 0
        
        for row_int in grid_ints:
            # Apply column flips to the row
            row_after_col_flips = row_int ^ mask
            
            # Count the number of 1s after column flips
            ones_count = 0
            temp = row_after_col_flips
            while temp:
                ones_count += temp & 1
                temp >>= 1
            
            # Decide whether to flip the row or not
            zeros_count = W - ones_count
            
            # Choose the option that minimizes the number of 1s
            min_ones = min(ones_count, zeros_count)
            total_sum += min_ones
        
        min_sum = min(min_sum, total_sum)
    
    return min_sum

# Read input
H, W = map(int, input().split())
grid_strings = []
for _ in range(H):
    grid_strings.append(input().strip())

# Calculate and print the answer
print(min_sum_after_operations(H, W, grid_strings))