# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    a, b, c, d, e, f = map(int, data[:6])
    g, h, i, j, k, l = map(int, data[6:])
    
    # Check overlap in x, y, z dimensions
    overlap_x = max(a, g) < min(d, j)
    overlap_y = max(b, h) < min(e, k)
    overlap_z = max(c, i) < min(f, l)
    
    if overlap_x and overlap_y and overlap_z:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()