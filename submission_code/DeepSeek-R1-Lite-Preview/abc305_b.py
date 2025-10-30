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

# Read input points p and q
p, q = input().split()

# Retrieve positions of p and q
pos_p = positions[p]
pos_q = positions[q]

# Calculate and print the distance
distance = abs(pos_p - pos_q)
print(distance)