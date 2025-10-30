def has_positive_volume(a, b, c, d, e, f, g, h, i, j, k, l):
    # Calculate the intersection coordinates
    x_overlap = max(0, min(d, j) - max(a, g))
    y_overlap = max(0, min(e, k) - max(b, h))
    z_overlap = max(0, min(f, l) - max(c, i))
    
    # If all overlaps are positive, then the intersection volume is positive
    return x_overlap > 0 and y_overlap > 0 and z_overlap > 0

import sys

# Read input
input_data = sys.stdin.read().strip().splitlines()
a, b, c, d, e, f = map(int, input_data[0].split())
g, h, i, j, k, l = map(int, input_data[1].split())

# Check for positive volume intersection
if has_positive_volume(a, b, c, d, e, f, g, h, i, j, k, l):
    print("Yes")
else:
    print("No")