# Read input
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check overlap in each dimension
x_overlap = max(a, g) < min(d, j)
y_overlap = max(b, h) < min(e, k)
z_overlap = max(c, i) < min(f, l)

# Determine if all three dimensions have positive overlap
print("Yes" if x_overlap and y_overlap and z_overlap else "No")