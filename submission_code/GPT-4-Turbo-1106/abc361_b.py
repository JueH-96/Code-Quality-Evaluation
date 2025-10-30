def check_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    # Check if there is no overlap on the x-axis
    if d <= g or j <= a:
        return False
    # Check if there is no overlap on the y-axis
    if e <= h or k <= b:
        return False
    # Check if there is no overlap on the z-axis
    if f <= i or l <= c:
        return False
    # If all checks passed, the cuboids intersect with positive volume
    return True

# Read input from stdin
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Check for intersection and print the result
if check_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    print("Yes")
else:
    print("No")