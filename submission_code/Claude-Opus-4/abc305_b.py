# YOUR CODE HERE
# Read input
p, q = input().split()

# Define the points and their positions
points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
point_to_index = {point: i for i, point in enumerate(points)}

# Define distances between adjacent points
distances = [3, 1, 4, 1, 5, 9]  # A-B, B-C, C-D, D-E, E-F, F-G

# Get indices of the two points
idx_p = point_to_index[p]
idx_q = point_to_index[q]

# Ensure we go from left to right
start = min(idx_p, idx_q)
end = max(idx_p, idx_q)

# Calculate total distance
total_distance = 0
for i in range(start, end):
    total_distance += distances[i]

print(total_distance)