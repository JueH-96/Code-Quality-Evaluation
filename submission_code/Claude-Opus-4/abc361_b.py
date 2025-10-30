# Read the first cuboid coordinates
a, b, c, d, e, f = map(int, input().split())

# Read the second cuboid coordinates
g, h, i, j, k, l = map(int, input().split())

# Check if there's positive overlap in x dimension
x_overlap = max(a, g) < min(d, j)

# Check if there's positive overlap in y dimension
y_overlap = max(b, h) < min(e, k)

# Check if there's positive overlap in z dimension
z_overlap = max(c, i) < min(f, l)

# The cuboids have positive volume intersection if and only if
# they have positive overlap in all three dimensions
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")