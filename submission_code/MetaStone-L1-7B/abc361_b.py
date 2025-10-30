# Read the input for the first cuboid
a, b, c, d, e, f = map(int, input().split())

# Read the input for the second cuboid
g, h, i, j, k, l = map(int, input().split())

# Check overlap on each axis
x_overlap = (a < j) and (g < d)
y_overlap = (b < k) and (h < e)
z_overlap = (c < l) and (i < f)

# Determine if all three axes have overlapping intervals
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")