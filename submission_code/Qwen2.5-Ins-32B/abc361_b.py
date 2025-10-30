import sys

def has_positive_volume_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    # Check if the cuboids do not overlap in any dimension
    if d <= g or a >= j or e <= h or b >= k or f <= i or c >= l:
        return False
    return True

# Read input from stdin
input_data = sys.stdin.read().strip().split()
a, b, c, d, e, f = map(int, input_data[:6])
g, h, i, j, k, l = map(int, input_data[6:])

# Determine if the intersection has a positive volume
if has_positive_volume_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    print("Yes")
else:
    print("No")