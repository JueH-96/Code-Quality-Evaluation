def main():
    # Read the first cuboid coordinates
    a, b, c, d, e, f = map(int, input().split())
    # Read the second cuboid coordinates
    g, h, i, j, k, l = map(int, input().split())
    
    # Check for overlap on each axis
    x_overlap = max(a, g) < min(d, j)
    y_overlap = max(b, h) < min(e, k)
    z_overlap = max(c, i) < min(f, l)
    
    # If all three overlaps are True, print "Yes"; else, "No"
    if x_overlap and y_overlap and z_overlap:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()