import sys

def main() -> None:
    # Read coordinates of the first cuboid
    a, b, c, d, e, f = map(int, sys.stdin.readline().split())
    # Read coordinates of the second cuboid
    g, h, i, j, k, l = map(int, sys.stdin.readline().split())

    # For a positive-volume intersection, the intervals on every axis
    # must overlap with positive length (strict inequality).
    overlap_x = max(a, g) < min(d, j)
    overlap_y = max(b, h) < min(e, k)
    overlap_z = max(c, i) < min(f, l)

    if overlap_x and overlap_y and overlap_z:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()