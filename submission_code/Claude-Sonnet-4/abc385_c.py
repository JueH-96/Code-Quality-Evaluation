n = int(input())
heights = list(map(int, input().split()))

max_buildings = 1

# Group positions by height
height_positions = {}
for i in range(n):
    h = heights[i]
    if h not in height_positions:
        height_positions[h] = []
    height_positions[h].append(i)

# For each height, find maximum arithmetic progression
for positions in height_positions.values():
    if len(positions) <= max_buildings:
        continue
    
    # Try all pairs as potential start of arithmetic progression
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # positions[i] and positions[j] define the arithmetic progression
            start = positions[i]
            diff = positions[j] - positions[i]
            count = 2
            next_pos = positions[j] + diff
            
            # Count how many more positions follow this pattern
            k = j + 1
            while k < len(positions):
                if positions[k] == next_pos:
                    count += 1
                    next_pos += diff
                k += 1
            
            max_buildings = max(max_buildings, count)

print(max_buildings)