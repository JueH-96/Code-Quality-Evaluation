# Create a dictionary mapping each point to its cumulative distance from A
distance = {
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

# Calculate and print the absolute difference
print(abs(distance[p] - distance[q]))