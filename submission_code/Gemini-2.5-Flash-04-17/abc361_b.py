import sys

# Read the coordinates of the first cuboid
line1 = sys.stdin.readline().split()
a, b, c, d, e, f = map(int, line1)

# Read the coordinates of the second cuboid
line2 = sys.stdin.readline().split()
g, h, i, j, k, l = map(int, line2)

# A cuboid C(min_x, min_y, min_z, max_x, max_y, max_z) is defined by the ranges [min_x, max_x], [min_y, max_y], [min_z, max_z].
# Given the problem constraints (e.g., a < d), (a,b,c) are the minimum coordinates and (d,e,f) are the maximum coordinates for the first cuboid.
# Similarly, (g,h,i) are the minimum coordinates and (j,k,l) are the maximum coordinates for the second cuboid.

# The intersection of two cuboids is non-empty and has positive volume
# if and only if the intersection of their ranges along each axis has positive length.
# The intersection range along the x-axis is [max(a, g), min(d, j)].
# This range has positive length if max(a, g) < min(d, j).

# Check for overlap along the x-axis
x_overlap = max(a, g) < min(d, j)

# Check for overlap along the y-axis
y_overlap = max(b, h) < min(e, k)

# Check for overlap along the z-axis
z_overlap = max(c, i) < min(f, l)

# The intersection has a positive volume if and only if there is a positive overlap along all three axes.
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")