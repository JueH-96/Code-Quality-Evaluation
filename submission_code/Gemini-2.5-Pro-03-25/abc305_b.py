# Define the coordinates of the points relative to A=0
# The points are A, B, C, D, E, F, G in order on a line.
# We can calculate the coordinate of each point by summing the adjacent distances starting from A=0.
point_coords = {
    'A': 0,
    'B': 3,      # Distance A-B = 3, so B is at 0 + 3 = 3
    'C': 3 + 1,  # Distance B-C = 1, so C is at 3 + 1 = 4
    'D': 4 + 4,  # Distance C-D = 4, so D is at 4 + 4 = 8
    'E': 8 + 1,  # Distance D-E = 1, so E is at 8 + 1 = 9
    'F': 9 + 5,  # Distance E-F = 5, so F is at 9 + 5 = 14
    'G': 14 + 9  # Distance F-G = 9, so G is at 14 + 9 = 23
}

# Read the input from stdin
# The input consists of two uppercase English letters p and q separated by a space.
# input() reads a line from stdin.
# .split() splits the string by whitespace into a list of strings.
# We unpack the list into variables p and q.
p, q = input().split()

# Look up the coordinates for the given points p and q using the dictionary.
coord_p = point_coords[p]
coord_q = point_coords[q]

# The distance between two points on a straight line is the absolute difference
# of their coordinates.
distance = abs(coord_p - coord_q)

# Print the calculated distance to stdout.
print(distance)