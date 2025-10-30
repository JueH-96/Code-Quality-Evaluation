# YOUR CODE HERE

# Read the inputs
p, q = input().split()

# Define the distances between the points
distances = {'AB': 3, 'BC': 1, 'CD': 4, 'DE': 1, 'EF': 5, 'FG': 9}

# Calculate the distance between the points
distance = sum(distances[f'{p}{q}'] for p, q in zip(p, q))

# Print the distance
print(distance)