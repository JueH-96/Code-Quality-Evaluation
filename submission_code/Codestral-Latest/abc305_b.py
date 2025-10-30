# YOUR CODE HERE
points = {'A': 0, 'B': 3, 'C': 4, 'D': 8, 'E': 12, 'F': 13, 'G': 18}

# Read the input
p, q = input().split()

# Calculate the distance
distance = abs(points[p] - points[q])

# Print the distance
print(distance)