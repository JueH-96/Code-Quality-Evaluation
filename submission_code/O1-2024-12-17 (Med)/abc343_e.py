def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    V1, V2, V3 = map(int, input_data)

    # Each cube has volume 7^3 = 343.
    # From the standard partition formulas for 3 "identical" cubes of side 7, a necessary condition is:
    #   V1 + 2*V2 + 3*V3 = 3 * 343 = 1029.
    #
    # If this fails, no arrangement can possibly work.
    if V1 + 2*V2 + 3*V3 != 1029:
        print("No")
        return

    # Also, no arrangement can have triple-overlap volume > 343 (the volume of a single cube).
    if V3 > 343:
        print("No")
        return

    # We will place:
    #   C1 at (0,0,0),
    #   C2 at (dx12, dy12, dz12),
    #   C3 at (dx13, dy13, dz13),
    # with dx, dy, dz each in [0..7].
    #
    # For two cubes offset by (dx,dy,dz), their intersection volume is:
    #   p = max(0, 7 - dx) * max(0, 7 - dy) * max(0, 7 - dz)
    #
    # The triple intersection of C1, C2, C3 is the intersection of
    #   [0,7] in x, [0,7] in y, [0,7] in z,
    #   [dx12, dx12+7], [dy12, dy12+7], [dz12, dz12+7],
    #   [dx13, dx13+7], [dy13, dy13+7], [dz13, dz13+7].
    # Its side-length in x is:
    #   lx = max(0, min(7, dx12+7, dx13+7) - max(0, dx12, dx13))
    # similarly for y, z.  The volume is lx*ly*lz.
    #
    # Once p12, p13, p23, and t (triple overlap) are known,
    #   S2 = p12 + p13 + p23 - 3*t
    #   S1 = 3*343 - 2*(p12 + p13 + p23) + 3*t
    #   S3 = t
    #
    # We compare (S1, S2, S3) to (V1, V2, V3).  If they match, we output a solution.

    # Brute force all dx,dy,dz in [0..7] for C2 and C3.
    # That is 8^6 = 262,144 iterations, which is usually feasible in Python if done carefully.
    #
    # Note: we do not need negative shifts because the intersection volume depends on absolute
    # offsets.  Shifting by +3 or -3 yields the same shape/volume.  Also, the problem only asks
    # for any valid solution within |coordinate| <= 100, so [0..7] suffices.

    for dx12 in range(8):
        for dy12 in range(8):
            for dz12 in range(8):
                # Intersection volume of C1 and C2
                p12 = (max(0, 7 - dx12) *
                       max(0, 7 - dy12) *
                       max(0, 7 - dz12))
                for dx13 in range(8):
                    for dy13 in range(8):
                        for dz13 in range(8):
                            # Intersection volume of C1 and C3
                            p13 = (max(0, 7 - dx13) *
                                   max(0, 7 - dy13) *
                                   max(0, 7 - dz13))
                            # Intersection volume of C2 and C3 = depends on their relative offset
                            dx23 = abs(dx13 - dx12)
                            dy23 = abs(dy13 - dy12)
                            dz23 = abs(dz13 - dz12)
                            p23 = (max(0, 7 - dx23) *
                                   max(0, 7 - dy23) *
                                   max(0, 7 - dz23))

                            # Triple intersection volume
                            lx = min(7, dx12+7, dx13+7) - max(0, dx12, dx13)
                            if lx < 0: lx = 0
                            ly = min(7, dy12+7, dy13+7) - max(0, dy12, dy13)
                            if ly < 0: ly = 0
                            lz = min(7, dz12+7, dz13+7) - max(0, dz12, dz13)
                            if lz < 0: lz = 0
                            t = lx * ly * lz  # triple intersection

                            if t != V3:
                                continue

                            # Exactly-two coverage
                            s2 = p12 + p13 + p23 - 3*t
                            if s2 != V2:
                                continue

                            # Exactly-one coverage
                            #   S1 = 3*343 - 2*(p12+p13+p23) + 3*t
                            # or equivalently 1029 - 2*s2 - 3*t
                            s1 = 1029 - 2*s2 - 3*t
                            if s1 == V1:
                                # Found a solution
                                print("Yes")
                                print(0, 0, 0,
                                      dx12, dy12, dz12,
                                      dx13, dy13, dz13)
                                return

    # If we exhaust loops with no match:
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()