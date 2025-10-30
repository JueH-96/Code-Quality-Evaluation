#!/usr/bin/env python3
import sys

def main():
    # Read the first cuboid coordinates
    a, b, c, d, e, f = map(int, sys.stdin.readline().split())
    # Read the second cuboid coordinates
    g, h, i, j, k, l = map(int, sys.stdin.readline().split())

    # Compute overlap length along each axis
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)

    # Check if all overlaps are strictly positive
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

# Call main to execute the solution
main()