# YOUR CODE HERE
def min_swaps_to_contiguous_ones(N, S):
    # Step 1: Find all positions of '1's
    ones_positions = [i for i, char in enumerate(S) if char == '1']
    
    # Step 2: Calculate the median position
    num_ones = len(ones_positions)
    median_index = num_ones // 2
    median_position = ones_positions[median_index]
    
    # Step 3: Calculate the minimum swaps needed
    swaps = 0
    for i in range(num_ones):
        # The target position for the i-th '1' in the sorted order
        target_position = median_position - median_index + i
        swaps += abs(ones_positions[i] - target_position)
    
    return swaps

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
S = data[1]

# Calculate and print the result
result = min_swaps_to_contiguous_ones(N, S)
print(result)