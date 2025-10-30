# Read the input points
p, q = input().split()

# Define the distances between each pair of adjacent points
distances = {'AB': 3, 'BC': 1, 'CD': 4, 'DE': 1, 'EF': 5, 'FG': 9}

# Define the order of the points
points_order = 'ABCDEFG'

# Find the positions of the input points
p_index = points_order.index(p)
q_index = points_order.index(q)

# Make sure p_index is less than q_index to simplify calculation
if p_index > q_index:
    p_index, q_index = q_index, p_index

# Calculate the distance between the points p and q
distance = 0
for i in range(p_index, q_index):
    distance += distances[points_order[i] + points_order[i + 1]]

# Print the distance
print(distance)