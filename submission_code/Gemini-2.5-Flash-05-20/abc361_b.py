import sys

def solve():
    # Read the coordinates for the first cuboid (C1)
    # C1 is defined by a diagonal connecting (a,b,c) and (d,e,f).
    # Given constraints like 'a < d', 'b < e', 'c < f',
    # (a,b,c) represents the minimum corner and (d,e,f) represents the maximum corner.
    line1_coords = list(map(int, sys.stdin.readline().split()))
    a, b, c, d, e, f = line1_coords

    # Read the coordinates for the second cuboid (C2)
    # C2 is defined by a diagonal connecting (g,h,i) and (j,k,l).
    # Similarly, (g,h,i) is the minimum corner and (j,k,l) is the maximum corner.
    line2_coords = list(map(int, sys.stdin.readline().split()))
    g, h, i, j, k, l = line2_coords

    # For two cuboids to have an intersection with positive volume,
    # they must have a positive length intersection along the X-axis,
    # AND a positive length intersection along the Y-axis,
    # AND a positive length intersection along the Z-axis.

    # 1. Check for positive overlap along the X-axis:
    # The intersection interval along X is [max(a, g), min(d, j)].
    # For positive length, max(a, g) must be strictly less than min(d, j).
    x_overlap = max(a, g) < min(d, j)

    # 2. Check for positive overlap along the Y-axis:
    # The intersection interval along Y is [max(b, h), min(e, k)].
    # For positive length, max(b, h) must be strictly less than min(e, k).
    y_overlap = max(b, h) < min(e, k)

    # 3. Check for positive overlap along the Z-axis:
    # The intersection interval along Z is [max(c, i), min(f, l)].
    # For positive length, max(c, i) must be strictly less than min(f, l).
    z_overlap = max(c, i) < min(f, l)

    # If all three dimensions have a positive length overlap,
    # then the overall intersection has a positive volume.
    if x_overlap and y_overlap and z_overlap:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

# Call the solve function to execute the logic
if __name__ == '__main__':
    solve()