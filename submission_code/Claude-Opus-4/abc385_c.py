# YOUR CODE HERE
n = int(input())
heights = list(map(int, input().split()))

max_buildings = 1  # At least one building can always be chosen

# Group buildings by height
height_positions = {}
for i in range(n):
    h = heights[i]
    if h not in height_positions:
        height_positions[h] = []
    height_positions[h].append(i)

# For each height, find the maximum number of buildings at equal intervals
for h, positions in height_positions.items():
    if len(positions) == 1:
        continue
    
    # Try all possible starting positions and intervals
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Calculate the interval
            interval = positions[j] - positions[i]
            
            # Count how many buildings follow this pattern
            count = 2  # We already have positions[i] and positions[j]
            last_pos = positions[j]
            
            for k in range(j + 1, len(positions)):
                if positions[k] - last_pos == interval:
                    count += 1
                    last_pos = positions[k]
            
            max_buildings = max(max_buildings, count)

print(max_buildings)