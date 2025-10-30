def check_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    # Check if the intersection has a positive volume
    # by checking if the projections of the cuboids on each axis overlap
    x_overlap = (max(a, g) <= min(d, j))
    y_overlap = (max(b, h) <= min(e, k))
    z_overlap = (max(c, i) <= min(f, l))
    
    # If all projections overlap, the intersection has a positive volume
    return x_overlap and y_overlap and z_overlap

# Read input from stdin
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check if the intersection has a positive volume
if check_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    print("Yes")
else:
    print("No")