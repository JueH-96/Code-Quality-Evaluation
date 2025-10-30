def find_max_buildings(N, heights):
    max_buildings = 1  # We can always choose 1 building
    
    # Try each possible height
    for height in set(heights):
        # Find positions of buildings with this height
        positions = [i for i, h in enumerate(heights) if h == height]
        
        # For each starting position
        for start_idx in range(len(positions)):
            # For each possible interval
            for interval in range(1, N):
                count = 1
                last_pos = positions[start_idx]
                
                # Try to find buildings at equal intervals with same height
                for pos in positions[start_idx + 1:]:
                    if pos == last_pos + interval:
                        count += 1
                        last_pos = pos
                
                max_buildings = max(max_buildings, count)
    
    return max_buildings

# Read input
N = int(input())
heights = list(map(int, input().split()))

# Print result
print(find_max_buildings(N, heights))