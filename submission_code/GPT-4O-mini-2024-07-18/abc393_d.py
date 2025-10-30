def min_swaps_to_contiguous_ones(N, S):
    # Find the positions of all '1's in the string
    positions = [i for i, char in enumerate(S) if char == '1']
    
    # The number of '1's
    count_ones = len(positions)
    
    # If there is only one '1', they are already contiguous
    if count_ones <= 1:
        return 0
    
    # Calculate the ideal positions for the '1's to be contiguous
    # The ideal positions would be centered around the median of the current positions
    median_index = count_ones // 2
    median_position = positions[median_index]
    
    # Calculate the minimum number of swaps needed
    swaps = 0
    for i in range(count_ones):
        # The ideal position for the i-th '1' is median_position - (median_index - i)
        ideal_position = median_position - (median_index - i)
        swaps += abs(positions[i] - ideal_position)
    
    return swaps

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
S = data[1]

# Get the result and print it
result = min_swaps_to_contiguous_ones(N, S)
print(result)