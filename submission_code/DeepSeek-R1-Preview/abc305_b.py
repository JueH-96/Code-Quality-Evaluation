# Read the positions of the points
positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Read input
p, q = input().split()

# Calculate the distance
distance = abs(positions[p] - positions[q])

# Output the result
print(distance)