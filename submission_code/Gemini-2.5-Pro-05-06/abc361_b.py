# Read coordinates for the first cuboid
a, b, c, d, e, f = map(int, input().split())

# Read coordinates for the second cuboid
g, h, i, j, k, l = map(int, input().split())

# Calculate the start and end of the intersection interval for the x-axis
# Cuboid 1 x-range: [a, d]
# Cuboid 2 x-range: [g, j]
# Intersection x-range: [max(a, g), min(d, j)]
x_overlap_start = max(a, g)
x_overlap_end = min(d, j)

# Calculate the start and end of the intersection interval for the y-axis
# Cuboid 1 y-range: [b, e]
# Cuboid 2 y-range: [h, k]
# Intersection y-range: [max(b, h), min(e, k)]
y_overlap_start = max(b, h)
y_overlap_end = min(e, k)

# Calculate the start and end of the intersection interval for the z-axis
# Cuboid 1 z-range: [c, f]
# Cuboid 2 z-range: [i, l]
# Intersection z-range: [max(c, i), min(f, l)]
z_overlap_start = max(c, i)
z_overlap_end = min(f, l)

# Check if the length of the intersection interval is positive for all three dimensions.
# A positive length means overlap_start < overlap_end.
# If overlap_start >= overlap_end, the length is zero or negative (no overlap).

has_positive_x_overlap = (x_overlap_start < x_overlap_end)
has_positive_y_overlap = (y_overlap_start < y_overlap_end)
has_positive_z_overlap = (z_overlap_start < z_overlap_end)

# The intersection has positive volume if and only if there's positive overlap in all three dimensions.
if has_positive_x_overlap and has_positive_y_overlap and has_positive_z_overlap:
    print("Yes")
else:
    print("No")