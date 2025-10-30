def main():
    import sys
    V1, V2, V3 = map(int, sys.stdin.readline().split())
    # necessary inclusion–exclusion check:
    if V1 + 2*V2 + 3*V3 != 3*7**3:
        print("No")
        return
    # brute force over positions
    for px in range(0, 8):
        lx12 = 7 - px
        for qx in range(px, 15):
            # overlap‑lengths in x
            if qx <= 7:
                lx13 = 7 - qx
            else:
                lx13 = 0
            dx = qx - px
            if dx <= 7:
                lx23 = 7 - dx
            else:
                lx23 = 0
            Lx123 = lx13  # chain: x1<=x2<=x3 => triple = ℓ13
            # y–axis
            for py in range(0, 8):
                ly12 = 7 - py
                for qy in range(py, 15):
                    if qy <= 7:
                        ly13 = 7 - qy
                    else:
                        ly13 = 0
                    dy = qy - py
                    if dy <= 7:
                        ly23 = 7 - dy
                    else:
                        ly23 = 0
                    Ly123 = ly13
                    # partial products for pairs in x,y
                    P12 = lx12 * ly12
                    P13 = lx13 * ly13
                    P23 = lx23 * ly23
                    P123 = Lx123 * Ly123
                    # z–axis
                    for pz in range(0, 8):
                        lz12 = 7 - pz
                        for qz in range(pz, 15):
                            if qz <= 7:
                                lz13 = 7 - qz
                            else:
                                lz13 = 0
                            dz = qz - pz
                            if dz <= 7:
                                lz23 = 7 - dz
                            else:
                                lz23 = 0
                            Lz123 = lz13
                            # compute triple intersection
                            T = P123 * Lz123
                            if T != V3:
                                continue
                            # pairwise
                            S12 = P12 * lz12
                            S13 = P13 * lz13
                            S23 = P23 * lz23
                            V2_calc = S12 + S13 + S23 - 3 * T
                            if V2_calc != V2:
                                continue
                            V1_calc = 3*343 - 2*V2_calc - 3*T
                            if V1_calc != V1:
                                continue
                            # success
                            print("Yes")
                            # cube1 at (0,0,0), cube2 at (px,py,pz), cube3 at (qx,qy,qz)
                            print(0, 0, 0, px, py, pz, qx, qy, qz)
                            return
    # no solution found
    print("No")

if __name__ == "__main__":
    main()