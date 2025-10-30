# YOUR CODE HERE
# Define the positions of the points on the line
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

# Calculate the absolute distance between the two points
distance = abs(positions[p] - positions[q])

# Print the result
print(distance)