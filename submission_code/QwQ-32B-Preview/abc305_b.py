# Define positions based on cumulative distances
positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Read input points p and q
p, q = input().strip().split()

# Get positions of p and q
pos_p = positions[p]
pos_q = positions[q]

# Calculate the distance as the absolute difference
distance = abs(pos_p - pos_q)

# Print the result
print(distance)