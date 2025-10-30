def min_sum_after_operations(H, W, grid):
    # We will calculate the minimum sum we can achieve for each column
    total_sum = 0
    
    # Iterate over each column
    for col in range(W):
        count_ones = sum(int(grid[row][col]) for row in range(H))
        count_zeros = H - count_ones
        
        # For each column, we can either keep the ones or flip them to zeros
        # The minimum number of 1s we can achieve in this column is the minimum of
        # the count of ones or the count of zeros after flipping
        total_sum += min(count_ones, count_zeros)
    
    return total_sum

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
grid = data[1:H+1]

# Calculate the result
result = min_sum_after_operations(H, W, grid)

# Print the result
print(result)