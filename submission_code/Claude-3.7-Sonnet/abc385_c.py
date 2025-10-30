def solve(N, heights):
    max_count = 1  # At minimum, we can choose 1 building
    
    # Group positions by height
    height_to_positions = {}
    for i in range(N):
        h = heights[i]
        if h not in height_to_positions:
            height_to_positions[h] = []
        height_to_positions[h].append(i)
    
    # For each height, find the maximum number of buildings we can choose
    for positions in height_to_positions.values():
        positions_set = set(positions)  # For O(1) lookup
        
        # For each starting position
        for i in range(len(positions)):
            # For each possible interval
            for j in range(i+1, len(positions)):
                interval = positions[j] - positions[i]
                count = 1  # Start counting from positions[i]
                current_pos = positions[i]
                
                # Check if all positions in the arithmetic sequence are present
                while current_pos + interval < N:
                    current_pos += interval
                    if current_pos in positions_set:
                        count += 1
                    else:
                        break
                
                max_count = max(max_count, count)
    
    return max_count

# Read input
N = int(input())
heights = list(map(int, input().split()))

# Print output
print(solve(N, heights))