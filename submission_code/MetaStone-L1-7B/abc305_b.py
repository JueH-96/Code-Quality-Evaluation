# Positions of each point relative to A (starting at 0)
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