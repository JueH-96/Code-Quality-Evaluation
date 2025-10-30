# Build the coordinates for each point
points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
distances = [3, 1, 4, 1, 5, 9]
coords = {points[0]: 0}
current = 0

for i in range(len(distances)):
    current += distances[i]
    coords[points[i + 1]] = current

# Read input values
p, q = input().split()

# Calculate the absolute difference between their coordinates
distance = abs(coords[p] - coords[q])

# Output the result
print(distance)