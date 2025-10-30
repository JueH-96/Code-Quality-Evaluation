# Read input
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Calculate overlapping intervals for each axis
x_overlap = max(a, g) < min(d, j)
y_overlap = max(b, h) < min(e, k)
z_overlap = max(c, i) < min(f, l)

# Check if all dimensions have positive overlap
print("Yes" if x_overlap and y_overlap and z_overlap else "No")