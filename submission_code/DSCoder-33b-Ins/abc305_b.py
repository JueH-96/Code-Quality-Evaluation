# YOUR CODE HERE

# Read the inputs
p, q = input().split()

# Define the distances between adjacent points
distances = {'A': 3, 'B': 1, 'C': 4, 'D': 1, 'E': 5, 'F': 9}

# Calculate the distance between the points p and q
distance = 0
for i in range(ord(p), ord(q)):
    distance += distances[chr(i)]

# Print the distance
print(distance)