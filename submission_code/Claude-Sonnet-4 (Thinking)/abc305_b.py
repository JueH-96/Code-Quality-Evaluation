# YOUR CODE HERE
# Read input
p, q = input().split()

# Define positions of each point on the line
positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Calculate distance between points p and q
distance = abs(positions[p] - positions[q])

# Output the result
print(distance)