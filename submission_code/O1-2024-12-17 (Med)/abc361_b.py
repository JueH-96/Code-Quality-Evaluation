def main():
    import sys

    # Read inputs
    data = list(map(int, sys.stdin.read().strip().split()))
    a, b, c, d, e, f = data[0], data[1], data[2], data[3], data[4], data[5]
    g, h, i, j, k, l = data[6], data[7], data[8], data[9], data[10], data[11]
    
    # Calculate overlap in each dimension
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)
    
    # Check if all overlaps are strictly positive
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

# Do not forget to call main
main()