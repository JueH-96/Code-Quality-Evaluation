# YOUR CODE HERE
import sys

def solve():
    """
    Solves the cuboid intersection problem by checking for positive overlap
    on all three axes.
    """
    # Read the coordinates of the first cuboid from standard input.
    # C(a,b,c,d,e,f) has a diagonal from (a,b,c) to (d,e,f).
    # The constraints 0 <= a < d, etc., imply that (a,b,c) is the
    # "min" corner and (d,e,f) is the "max" corner.
    # So, the x, y, and z ranges for the first cuboid are [a, d], [b, e], and [c, f].
    try:
        line1 = sys.stdin.readline().split()
        if not line1: return
        a, b, c, d, e, f = map(int, line1)

        # Read the coordinates of the second cuboid.
        # Its x, y, and z ranges are [g, j], [h, k], and [i, l].
        line2 = sys.stdin.readline().split()
        if not line2: return
        g, h, i, j, k, l = map(int, line2)
    except (IOError, ValueError):
        # This handles potential malformed or empty input lines,
        # which is good practice but not expected given the problem constraints.
        return

    # Two cuboids intersect with a positive volume if and only if their
    # projections onto each of the x, y, and z axes overlap with a positive length.

    # The intersection of two intervals [x1, x2] and [x3, x4] is the interval
    # [max(x1, x3), min(x2, x4)].
    # For this intersection interval to have a positive length, its end must be
    # strictly greater than its start. That is, min(x2, x4) > max(x1, x3).

    # We check this condition for all three axes.
    if (max(a, g) < min(d, j) and   # Check for positive length overlap on x-axis
        max(b, h) < min(e, k) and   # Check for positive length overlap on y-axis
        max(c, i) < min(f, l)):     # Check for positive length overlap on z-axis
        # If and only if all three axes have a positive-length overlap,
        # the intersection forms a cuboid with positive volume.
        print("Yes")
    else:
        # If any axis has zero or no overlap, the intersection is a plane,
        # a line, a point, or empty. In all these cases, the volume is not positive (it's 0).
        print("No")

solve()