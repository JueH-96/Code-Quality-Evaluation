# Define the points and their corresponding distances
points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
distances = [3, 1, 4, 1, 5, 9]

# Create a dictionary to store the cumulative distances
cumulative_distances = {points[0]: 0}
for i in range(1, len(points)):
    cumulative_distances[points[i]] = cumulative_distances[points[i-1]] + distances[i-1]

# Read the input points
p, q = input().split()

# Calculate the distance between the points
if points.index(p) < points.index(q):
    distance = cumulative_distances[q] - cumulative_distances[p]
else:
    distance = cumulative_distances[p] - cumulative_distances[q]

# Print the distance
print(distance)