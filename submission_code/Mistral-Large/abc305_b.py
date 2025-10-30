import sys

# Define the distances between adjacent points
distances = {
    ('A', 'B'): 3,
    ('B', 'C'): 1,
    ('C', 'D'): 4,
    ('D', 'E'): 1,
    ('E', 'F'): 5,
    ('F', 'G'): 9
}

# Define the order of points
points_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Read input
p, q = sys.stdin.read().strip().split()

# Ensure p comes before q in the order
if points_order.index(p) > points_order.index(q):
    p, q = q, p

# Calculate the distance
distance = 0
for i in range(points_order.index(p), points_order.index(q)):
    distance += distances[(points_order[i], points_order[i + 1])]

# Print the distance
print(distance)