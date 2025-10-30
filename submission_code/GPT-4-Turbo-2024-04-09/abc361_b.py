# Reading input values
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

# Calculate the overlap in each dimension
overlap_x = max(0, min(d, j) - max(a, g))
overlap_y = max(0, min(e, k) - max(b, h))
overlap_z = max(0, min(f, l) - max(c, i))

# Calculate the volume of the intersection
intersection_volume = overlap_x * overlap_y * overlap_z

# Output whether the intersection volume is positive
if intersection_volume > 0:
    print("Yes")
else:
    print("No")