def main():
    import sys
    data = sys.stdin.read().split()
    # Parse the 12 integers
    a, b, c, d, e, f = map(int, data[:6])
    g, h, i, j, k, l = map(int, data[6:12])

    # Compute the overlap along each axis
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)

    # Positive overlap on all three axes means positive volume intersection
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

# Call main to execute
main()