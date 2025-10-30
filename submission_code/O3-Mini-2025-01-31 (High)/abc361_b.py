def main():
    import sys
    
    data = sys.stdin.read().split()
    # Convert all input values to integers.
    values = list(map(int, data))
    
    # Unpack the inputs: first cuboid is defined by a, b, c, d, e, f.
    # Second cuboid is defined by g, h, i, j, k, l.
    a, b, c, d, e, f = values[0:6]
    g, h, i, j, k, l = values[6:12]
    
    # Compute the overlap intervals along each axis.
    # For a positive overlap along an axis: max(lower_bounds) < min(upper_bounds)
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)
    
    # Check if each axis has a positive length of intersection.
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()