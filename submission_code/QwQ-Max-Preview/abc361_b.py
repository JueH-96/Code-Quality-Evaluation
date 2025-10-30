# Read the input values for the two cuboids
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Calculate the overlapping ranges for each axis
x_start = max(a, g)
x_end = min(d, j)
y_start = max(b, h)
y_end = min(e, k)
z_start = max(c, i)
z_end = min(f, l)

# Check if all dimensions have a positive overlap
if x_end > x_start and y_end > y_start and z_end > z_start:
    print("Yes")
else:
    print("No")