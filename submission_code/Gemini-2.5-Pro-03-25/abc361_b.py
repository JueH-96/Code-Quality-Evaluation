# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    """
    Reads the coordinates of two cuboids and determines if their intersection
    has a positive volume.
    """
    # Read coordinates for the first cuboid from standard input
    line1 = list(map(int, sys.stdin.readline().split()))
    
    # Unpack coordinates: (a,b,c) is one corner, (d,e,f) is the opposite corner.
    # The constraints (a<d, b<e, c<f) ensure that (a,b,c) is the minimum corner
    # and (d,e,f) is the maximum corner.
    # The cuboid spans the coordinate ranges [a, d), [b, e), [c, f).
    a, b, c, d, e, f = line1[0], line1[1], line1[2], line1[3], line1[4], line1[5]

    # Read coordinates for the second cuboid from standard input
    line2 = list(map(int, sys.stdin.readline().split()))
    
    # Unpack coordinates: (g,h,i) is one corner, (j,k,l) is the opposite corner.
    # The constraints (g<j, h<k, i<l) ensure that (g,h,i) is the minimum corner
    # and (j,k,l) is the maximum corner.
    # The cuboid spans the coordinate ranges [g, j), [h, k), [i, l).
    g, h, i, j, k, l = line2[0], line2[1], line2[2], line2[3], line2[4], line2[5]

    # To determine if the intersection volume is positive, we need to check if
    # the intersection has positive length along each of the three dimensions (x, y, z).

    # Check for positive overlap in the x-dimension
    # The intervals are [a, d) and [g, j).
    # Positive overlap occurs if and only if the start of one interval is strictly
    # less than the end of the other interval, for both pairs.
    # Condition: a < j AND g < d
    x_overlap_positive = (a < j) and (g < d)

    # Check for positive overlap in the y-dimension
    # The intervals are [b, e) and [h, k).
    # Condition: b < k AND h < e
    y_overlap_positive = (b < k) and (h < e)

    # Check for positive overlap in the z-dimension
    # The intervals are [c, f) and [i, l).
    # Condition: c < l AND i < f
    z_overlap_positive = (c < l) and (i < f)

    # The intersection of the two cuboids has a positive volume if and only if
    # there is a positive overlap along all three dimensions (x, y, and z).
    if x_overlap_positive and y_overlap_positive and z_overlap_positive:
        # If all dimensions have positive overlap, the intersection volume is positive.
        print("Yes")
    else:
        # Otherwise, the intersection volume is zero (e.g., touching at a face, edge, or vertex)
        # or the intersection is empty (cuboids are separate or disjoint). In either case,
        # the volume is not positive.
        print("No")

# Execute the solve function when the script is run
solve()