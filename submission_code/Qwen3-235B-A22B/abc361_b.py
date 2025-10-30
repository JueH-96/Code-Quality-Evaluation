def main():
    # Read input
    a, b, c, d, e, f = map(int, input().split())
    g, h, i, j, k, l = map(int, input().split())
    
    # Calculate overlap in x-axis
    x_start = max(a, g)
    x_end = min(d, j)
    x_overlap = x_start < x_end
    
    # Calculate overlap in y-axis
    y_start = max(b, h)
    y_end = min(e, k)
    y_overlap = y_start < y_end
    
    # Calculate overlap in z-axis
    z_start = max(c, i)
    z_end = min(f, l)
    z_overlap = z_start < z_end
    
    # Check if all overlaps are positive
    if x_overlap and y_overlap and z_overlap:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()