a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Calculate overlap along each axis
x_overlap = min(d, j) - max(a, g)
y_overlap = min(e, k) - max(b, h)
z_overlap = min(f, l) - max(c, i)

# Check if all overlaps are positive
if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
    print("Yes")
else:
    print("No")