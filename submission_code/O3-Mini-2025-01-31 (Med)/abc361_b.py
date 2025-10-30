def main():
    import sys
    data = sys.stdin.read().split()
    # There should be 12 integers
    vals = list(map(int, data))
    if len(vals) != 12:
        return
    
    a, b, c, d, e, f = vals[0:6]
    g, h, i, j, k, l = vals[6:12]

    # Find overlapping interval in x, y, z dimensions.
    # For a positive volume, the length of intersection in each dimension must be > 0.
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)
    
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()