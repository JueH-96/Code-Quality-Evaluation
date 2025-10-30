# Read the input values for the two cuboids
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Calculate the overlap in each dimension
x_overlap = max(a, g) < min(d, j)
y_overlap = max(b, h) < min(e, k)
z_overlap = max(c, i) < min(f, l)

# Check if all three dimensions have a positive overlap
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")