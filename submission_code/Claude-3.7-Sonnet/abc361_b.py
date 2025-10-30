# Read the coordinates of the two cuboids
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# For intersection to have positive volume, there must be overlap in all three dimensions
# Check if there's overlap in x direction
x_overlap = max(a, g) < min(d, j)

# Check if there's overlap in y direction
y_overlap = max(b, h) < min(e, k)

# Check if there's overlap in z direction
z_overlap = max(c, i) < min(f, l)

# All three dimensions must have positive overlap for a positive volume
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")