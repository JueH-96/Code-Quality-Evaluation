# Read input
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check if there is overlap in each dimension
x_overlap = not (d <= g or j <= a)
y_overlap = not (e <= h or k <= b) 
z_overlap = not (f <= i or l <= c)

# If there is overlap in all dimensions, intersection has positive volume
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")