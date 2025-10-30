def main():
    import sys
    V1, V2, V3 = map(int, sys.stdin.readline().split())
    # We arrange cubes C1 at (0,0,0), C2 at (d,0,0), C3 at (0,d,0), side length = 7.
    # Compute the resulting:
    #   V3 = 7*(7-d)^2,
    #   V2 = 14*d*(7-d),
    #   V1 = 7*d*(d+14).
    # Try integer d in [0..7].
    for d in range(8):
        v3 = 7 * (7 - d) * (7 - d)
        v2 = 14 * d * (7 - d)
        v1 = 7 * d * (d + 14)
        if v1 == V1 and v2 == V2 and v3 == V3:
            print("Yes")
            # C1: (0,0,0)
            # C2: (d,0,0)
            # C3: (0,d,0)
            print(0, 0, 0, d, 0, 0, 0, d, 0)
            return
    print("No")

if __name__ == "__main__":
    main()