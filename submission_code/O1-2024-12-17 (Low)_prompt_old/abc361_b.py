def solve():
    import sys
    data = sys.stdin.read().strip().split()
    a, b, c, d, e, f = map(int, data[:6])
    g, h, i, j, k, l = map(int, data[6:])

    # Check for overlap along each dimension.
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)

    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

# Call the solve() function
solve()