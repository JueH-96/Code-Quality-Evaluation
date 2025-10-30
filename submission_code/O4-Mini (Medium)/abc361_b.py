import sys

def main():
    # Read input
    a, b, c, d, e, f = map(int, sys.stdin.readline().split())
    g, h, i, j, k, l = map(int, sys.stdin.readline().split())

    # Compute overlap along each axis
    overlap_x = min(d, j) - max(a, g)
    overlap_y = min(e, k) - max(b, h)
    overlap_z = min(f, l) - max(c, i)

    # Positive volume intersection if all overlaps > 0
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()