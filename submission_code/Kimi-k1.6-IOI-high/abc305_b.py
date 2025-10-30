# Create a dictionary to store the cumulative distance of each point from point A
positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Read the input points
p, q = input().split()

# Calculate the absolute difference between their positions
distance = abs(positions[p] - positions[q])

# Output the result
print(distance)