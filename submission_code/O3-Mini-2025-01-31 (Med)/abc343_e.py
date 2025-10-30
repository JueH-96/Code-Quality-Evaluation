def main():
    import sys
    input_line = sys.stdin.read().strip().split()
    if not input_line:
        return
    V1, V2, V3 = map(int, input_line)
    total = 3 * 7**3  # 3*343 = 1029
    # Necessary condition: V1 + 2*V2 + 3*V3 must be 1029.
    if V1 + 2*V2 + 3*V3 != total:
        sys.stdout.write("No")
        return

    # First try a symmetric construction.
    # If we set (with u = 7 - d):
    #   C1 = (0,0,0)
    #   C2 = (0, 7 - u, 0)
    #   C3 = (7 - u, 0, 0)
    # then it is easy to check that:
    #   V3 = 7*u*u,  V2 = 98*u - 14*u*u,  V1 = 1029 - 196*u + 7*u*u.
    for u in range(0, 8):
        if 7*u*u == V3 and (98*u - 14*u*u) == V2 and (1029 - 196*u + 7*u*u) == V1:
            d = 7 - u
            # Found valid configuration:
            out = ["Yes"]
            out.append("0 0 0 {} {} {} {} {} {}".format(0, d, 0, d, 0, 0))
            sys.stdout.write("
".join(out))
            return

    # If the symmetric configuration did not work then try a general search.
    # We fix the cubes to be aligned in every dimension.
    # In each dimension, we set cube1's interval as [0,7],
    # cube2's as [a, a+7] and cube3's as [b, b+7].
    # (It is enough to search a,b in the range -7..7.)
    def overlap(s1, e1, s2, e2):
        return max(0, min(e1,e2) - max(s1,s2))
    def overlap3(s1, e1, s2, e2, s3, e3):
        return max(0, min(e1,e2,e3) - max(s1,s2,s3))
    
    # Build candidate list for one dimension.
    # Each candidate is (a, b, L12, L13, L23, t)
    # where:
    #   L12 = overlap(cube1, cube2)
    #   L13 = overlap(cube1, cube3)
    #   L23 = overlap(cube2, cube3)
    #   t   = triple overlap = overlap(cube1, cube2, cube3)
    cand = []
    for a in range(-7, 8):
        for b in range(-7, 8):
            L12 = overlap(0, 7, a, a+7)
            L13 = overlap(0, 7, b, b+7)
            L23 = overlap(a, a+7, b, b+7)
            t = overlap3(0, 7, a, a+7, b, b+7)
            cand.append((a, b, L12, L13, L23, t))
    # For a given dimension, if the candidate (a, b, L12, L13, L23, t) is chosen,
    # then the cube’s coordinate in that axis becomes:
    # cube1 = 0, cube2 = a, cube3 = b.
    #
    # In 3d the overall volumes are computed multiplicatively.
    # Let the candidate in x, y and z be (ax, bx, L12_x, L13_x, L23_x, t_x),
    # (ay, by, L12_y, L13_y, L23_y, t_y) and (az, bz, L12_z, L13_z, L23_z, t_z)
    # Then:
    #   Triple intersection volume: T = t_x * t_y * t_z.
    #   For each pair, e.g. P12 = L12_x * L12_y * L12_z (and similarly for P13, P23).
    # And by inclusion–exclusion, one may show that
    #   V2 = (P12 + P13 + P23) - 3*T  and  V1 = 1029 - 2*(P12+P13+P23) + 3*T.
    target = V2 + 3*V3
    n = len(cand)
    for cx in cand:
        for cy in cand:
            for cz in cand:
                T = cx[5] * cy[5] * cz[5]
                if T != V3:
                    continue
                P12 = cx[2] * cy[2] * cz[2]
                P13 = cx[3] * cy[3] * cz[3]
                P23 = cx[4] * cy[4] * cz[4]
                if P12 + P13 + P23 != target:
                    continue
                # Build the 3d cube coordinates:
                # x–axis: cube1 = 0, cube2 = cx[0], cube3 = cx[1]
                # y–axis: cube1 = 0, cube2 = cy[0], cube3 = cy[1]
                # z–axis: cube1 = 0, cube2 = cz[0], cube3 = cz[1]
                a1 = (0, 0, 0)
                a2 = (cx[0], cy[0], cz[0])
                a3 = (cx[1], cy[1], cz[1])
                out = ["Yes"]
                out.append("{} {} {} {} {} {} {} {} {}".format(
                    a1[0], a1[1], a1[2],
                    a2[0], a2[1], a2[2],
                    a3[0], a3[1], a3[2]
                ))
                sys.stdout.write("
".join(out))
                return
    sys.stdout.write("No")

if __name__ == '__main__':
    main()