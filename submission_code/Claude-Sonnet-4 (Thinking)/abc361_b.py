# Read input
line1 = input().split()
line2 = input().split()

a, b, c, d, e, f = map(int, line1)
g, h, i, j, k, l = map(int, line2)

# Check intersection in all three dimensions
x_intersect = max(a, g) < min(d, j)
y_intersect = max(b, h) < min(e, k)
z_intersect = max(c, i) < min(f, l)

if x_intersect and y_intersect and z_intersect:
    print("Yes")
else:
    print("No")