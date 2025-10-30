# YOUR CODE HERE

# Distances between adjacent points as given in the problem statement
# A-B: 3
# B-C: 1
# C-D: 4
# D-E: 1
# E-F: 5
# F-G: 9
adjacent_distances = [3, 1, 4, 1, 5, 9]

# Names of the points in their order on the line
point_names_ordered = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Calculate absolute coordinates for each point, assuming A is at 0
# We'll store these in a dictionary: {'PointName': coordinate}
absolute_coordinates = {}
current_coordinate = 0

# The first point (A) is at coordinate 0
absolute_coordinates[point_names_ordered[0]] = current_coordinate

# Calculate coordinates for the remaining points
for i in range(len(adjacent_distances)):
    # Add the distance to the next point
    current_coordinate += adjacent_distances[i]
    # Store the coordinate of the next point
    absolute_coordinates[point_names_ordered[i+1]] = current_coordinate

# Read the two point names from standard input
p_name, q_name = input().split()

# Get the coordinates of the two points from our pre-calculated dictionary
coord_p = absolute_coordinates[p_name]
coord_q = absolute_coordinates[q_name]

# The distance between two points on a line is the absolute difference of their coordinates
distance = abs(coord_p - coord_q)

# Print the result
print(distance)