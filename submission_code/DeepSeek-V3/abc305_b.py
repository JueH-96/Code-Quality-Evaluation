# YOUR CODE HERE
# Read the input
p, q = input().split()

# Define the positions of the points
positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Calculate the absolute difference
distance = abs(positions[p] - positions[q])

# Print the result
print(distance)