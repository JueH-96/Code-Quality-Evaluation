# Read the order of points and the distances between them
points_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
distances = [3, 1, 4, 1, 5, 9]

# Calculate cumulative positions starting from A (0)
positions = [0]
current = 0
for d in distances:
    current += d
    positions.append(current)

# Create a dictionary to map each point to its position
position_map = {point: pos for point, pos in zip(points_order, positions)}

# Read input
p, q = input().split()

# Calculate and print the absolute distance
distance = abs(position_map[p] - position_map[q])
print(distance)