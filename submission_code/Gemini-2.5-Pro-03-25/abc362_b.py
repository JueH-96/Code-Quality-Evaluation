# YOUR CODE HERE
import sys

# Read coordinates function for cleaner input reading
def read_coords():
    """Reads a line from stdin and returns two integers."""
    return map(int, sys.stdin.readline().split())

# Read the coordinates of the three points A, B, and C
xa, ya = read_coords()
xb, yb = read_coords()
xc, yc = read_coords()

# Function to calculate the square of the distance between two points
def dist_sq(x1, y1, x2, y2):
    """Calculates the square of the Euclidean distance between (x1, y1) and (x2, y2)."""
    dx = x1 - x2
    dy = y1 - y2
    return dx*dx + dy*dy # Use squared distance to avoid floating-point numbers

# Calculate the square of the lengths of the three sides of the triangle ABC
# Side AB squared
d_ab_sq = dist_sq(xa, ya, xb, yb)
# Side BC squared
d_bc_sq = dist_sq(xb, yb, xc, yc)
# Side CA squared
d_ca_sq = dist_sq(xc, yc, xa, ya)

# Check if the Pythagorean theorem holds for any combination of the sides.
# A triangle is a right triangle if the square of one side is equal to the sum
# of the squares of the other two sides.

# Check if angle at B is right (AC is hypotenuse)
condition1 = (d_ab_sq + d_bc_sq == d_ca_sq)
# Check if angle at C is right (AB is hypotenuse)
condition2 = (d_bc_sq + d_ca_sq == d_ab_sq)
# Check if angle at A is right (BC is hypotenuse)
condition3 = (d_ca_sq + d_ab_sq == d_bc_sq)

# If any of the conditions are true, the triangle is a right triangle
if condition1 or condition2 or condition3:
    print("Yes")
else:
    print("No")

# END OF YOUR CODE HERE