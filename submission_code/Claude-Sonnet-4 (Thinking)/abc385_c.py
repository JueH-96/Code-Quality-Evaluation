n = int(input())
heights = list(map(int, input().split()))

# Group positions by height
height_to_positions = {}
for i, h in enumerate(heights):
    if h not in height_to_positions:
        height_to_positions[h] = []
    height_to_positions[h].append(i)

max_buildings = 1

for h, positions in height_to_positions.items():
    m = len(positions)
    position_set = set(positions)
    
    # Find longest arithmetic progression in positions
    max_for_this_height = 1
    
    for i in range(m):
        for j in range(i + 1, m):
            # Common difference
            d = positions[j] - positions[i]
            # Count how many terms in the AP starting from positions[i] with difference d
            count = 2  # We already have positions[i] and positions[j]
            current = positions[j] + d
            while current in position_set:
                count += 1
                current += d
            max_for_this_height = max(max_for_this_height, count)
    
    max_buildings = max(max_buildings, max_for_this_height)

print(max_buildings)