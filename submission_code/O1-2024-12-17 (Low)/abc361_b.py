def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    a, b, c, d, e, f, g, h, i, j, k, l = map(int, data)

    # Check overlap along X-axis
    overlap_x = min(d, j) - max(a, g)
    # Check overlap along Y-axis
    overlap_y = min(e, k) - max(b, h)
    # Check overlap along Z-axis
    overlap_z = min(f, l) - max(c, i)

    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

# Do not forget to call main!
main()