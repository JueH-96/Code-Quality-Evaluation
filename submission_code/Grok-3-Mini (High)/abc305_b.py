# Define the positions of each point
positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Read input from stdin
p, q = input().split()

# Calculate the distance using absolute difference of positions
distance = abs(positions[p] - positions[q])

# Output the distance to stdout
print(distance)