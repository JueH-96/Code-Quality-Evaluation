# a b c d e f
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check if there is an intersection in each dimension
x_intersect = max(a, g) < min(d, j)
y_intersect = max(b, h) < min(e, k)
z_intersect = max(c, i) < min(f, l)

# If there is an intersection in all three dimensions, the volume is positive
if x_intersect and y_intersect and z_intersect:
    print("Yes")
else:
    print("No")