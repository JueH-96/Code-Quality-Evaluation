# YOUR CODE HERE
def max_decorated_buildings(N, heights):
    max_count = 1  # At least one building can always be chosen
    
    # Iterate over each unique height
    for height in set(heights):
        # Collect all indices where the building has the current height
        indices = [i for i in range(N) if heights[i] == height]
        
        # Check for each possible interval
        for interval in range(1, N):
            # Try to form a sequence starting from each index
            for start in indices:
                count = 1
                current = start
                while True:
                    next_index = current + interval
                    if next_index < N and heights[next_index] == height:
                        count += 1
                        current = next_index
                    else:
                        break
                max_count = max(max_count, count)
    
    return max_count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
heights = list(map(int, data[1:]))

print(max_decorated_buildings(N, heights))