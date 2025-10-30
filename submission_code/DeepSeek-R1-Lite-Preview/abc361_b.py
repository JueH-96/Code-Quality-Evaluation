# Read input for the first cuboid
a, b, c, d, e, f = map(int, input().split())

# Read input for the second cuboid
g, h, i, j, k, l = map(int, input().split())

# Check for overlap in each dimension
# X-axis overlap: a < j and g < d
# Y-axis overlap: b < k and h < e
# Z-axis overlap: c < l and i < f
if a < j and g < d and b < k and h < e and c < l and i < f:
    print("Yes")
else:
    print("No")