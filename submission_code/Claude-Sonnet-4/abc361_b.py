# YOUR CODE HERE
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check if intervals overlap in all three dimensions
x_overlap = max(a, g) < min(d, j)
y_overlap = max(b, h) < min(e, k)
z_overlap = max(c, i) < min(f, l)

if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")