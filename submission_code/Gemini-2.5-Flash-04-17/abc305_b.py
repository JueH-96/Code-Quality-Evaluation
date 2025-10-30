# YOUR CODE HERE
import sys

# Read the two point names from standard input.
# The input is expected to be two uppercase English letters (A-G) separated by a space.
p, q = sys.stdin.readline().split()

# Define the positions of the points on the line relative to point A at position 0.
# The distances between adjacent points are given as:
# A-B: 3
# B-C: 1
# C-D: 4
# D-E: 1
# E-F: 5
# F-G: 9
# We compute the absolute position for each point by summing the distances from A:
# Position of A = 0
# Position of B = Pos(A) + distance(A,B) = 0 + 3 = 3
# Position of C = Pos(B) + distance(B,C) = 3 + 1 = 4
# Position of D = Pos(C) + distance(C,D) = 4 + 4 = 8
# Position of E = Pos(D) + distance(D,E) = 8 + 1 = 9
# Position of F = Pos(E) + distance(E,F) = 9 + 5 = 14
# Position of G = Pos(F) + distance(F,G) = 14 + 9 = 23
# A dictionary maps each point name to its absolute position.
point_positions = {
    'A': 0,
    'B': 3,
    'C': 4,
    'D': 8,
    'E': 9,
    'F': 14,
    'G': 23
}

# Look up the positions (coordinates) of the two input points p and q using the dictionary.
pos_p = point_positions[p]
pos_q = point_positions[q]

# The distance between two points on a line is the absolute difference of their positions (coordinates).
distance = abs(pos_p - pos_q)

# Print the calculated distance to standard output.
print(distance)