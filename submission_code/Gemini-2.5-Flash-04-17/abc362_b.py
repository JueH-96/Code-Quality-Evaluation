# YOUR CODE HERE
import sys

# Read coordinates of point A
line_a = sys.stdin.readline().split()
xA, yA = int(line_a[0]), int(line_a[1])

# Read coordinates of point B
line_b = sys.stdin.readline().split()
xB, yB = int(line_b[0]), int(line_b[1])

# Read coordinates of point C
line_c = sys.stdin.readline().split()
xC, yC = int(line_c[0]), int(line_c[1])

# Calculate squared distances between the points
# The squared distance between (x1, y1) and (x2, y2) is (x2 - x1)^2 + (y2 - y1)^2
dist_sq_AB = (xB - xA)**2 + (yB - yA)**2
dist_sq_BC = (xC - xB)**2 + (yC - yB)**2
dist_sq_CA = (xA - xC)**2 + (yA - yC)**2

# Store the squared distances in a list
sides_sq = [dist_sq_AB, dist_sq_BC, dist_sq_CA]

# Sort the squared distances in ascending order
# This allows us to easily identify the two potentially shorter sides and the potentially longest side
sides_sq.sort()

# Check if the triangle is a right triangle using the Pythagorean theorem
# For a right triangle with side lengths a, b, and c (where c is the hypotenuse), a^2 + b^2 = c^2
# Since we have squared distances, we check if the sum of the two smallest squared distances
# equals the largest squared distance.
# Using squared distances avoids potential floating-point inaccuracies that might arise
# from calculating square roots of the distances themselves.
# Also, since the points are not collinear, all squared distances are guaranteed to be positive.
if sides_sq[0] + sides_sq[1] == sides_sq[2]:
    print("Yes")
else:
    print("No")