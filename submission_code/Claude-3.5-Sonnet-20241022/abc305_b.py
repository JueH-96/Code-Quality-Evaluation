# YOUR CODE HERE
p, q = input().split()

# Create a dictionary to store distances between adjacent points
distances = {
    ('A', 'B'): 3,
    ('B', 'C'): 1,
    ('C', 'D'): 4,
    ('D', 'E'): 1,
    ('E', 'F'): 5,
    ('F', 'G'): 9
}

# Create a list of points in order
points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Get indices of input points
start = min(points.index(p), points.index(q))
end = max(points.index(p), points.index(q))

# Calculate total distance
total = 0
for i in range(start, end):
    total += distances[(points[i], points[i+1])]

print(total)