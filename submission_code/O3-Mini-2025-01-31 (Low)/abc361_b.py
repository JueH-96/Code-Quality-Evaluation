def main():
    import sys
    data = sys.stdin.read().split()
    data = list(map(int, data))
    
    # Unpack the first cuboid: C(a, b, c, d, e, f)
    a, b, c, d, e, f = data[0:6]
    # Unpack the second cuboid: C(g, h, i, j, k, l)
    g, h, i, j, k, l = data[6:12]
    
    # Calculate the overlapping region dimensions along each axis.
    x_overlap = min(d, j) - max(a, g)
    y_overlap = min(e, k) - max(b, h)
    z_overlap = min(f, l) - max(c, i)
    
    # Check if each overlap is strictly positive.
    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
        print("Yes")
    else:
        print("No")
    
if __name__ == '__main__':
    main()