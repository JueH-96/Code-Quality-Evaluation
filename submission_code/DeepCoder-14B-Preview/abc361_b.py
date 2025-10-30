# Read the input values for the first cuboid
a, b, c, d, e, f = map(int, input().split())

# Read the input values for the second cuboid
g, h, i, j, k, l = map(int, input().split())

# Check for overlap along each axis
x_overlap = (a < j) and (g < d)
y_overlap = (b < k) and (h < e)
z_overlap = (c < l) and (i < f)

# Determine if all axes have overlap
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")