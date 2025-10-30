# YOUR CODE HERE
# Define cumulative distances from point A
distances = {
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

# Calculate distance between p and q
result = abs(distances[p] - distances[q])

# Print result
print(result)