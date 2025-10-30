import sys

S = 7           # side length
VOL = S ** 3    # volume of one cube = 343


def overlap(d: int) -> int:
    """return overlap length when two segments of length 7 are shifted by d"""
    return S - d if d < S else 0


def main() -> None:
    V1, V2, V3 = map(int, sys.stdin.readline().split())

    # necessary (and obviously sufficient) relation among the three numbers
    if V1 + 2 * V2 + 3 * V3 != 3 * VOL:
        print("No")
        return

    # pre-compute overlaps for every possible shift (0 … 7)
    olap = [overlap(d) for d in range(S + 1)]

    # cube 1 is fixed at (0,0,0); cubes 2 and 3 are shifted by 0 … 7 units
    for dx2 in range(8):
        l12x = olap[dx2]
        for dy2 in range(8):
            l12y = olap[dy2]
            for dz2 in range(8):
                l12z = olap[dz2]
                x12 = l12x * l12y * l12z                    # |C1 ∩ C2|

                for dx3 in range(8):
                    l13x = olap[dx3]
                    l23x = olap[abs(dx2 - dx3)]
                    tx   = olap[max(dx2, dx3)]               # overlap of all 3 along x

                    for dy3 in range(8):
                        l13y = olap[dy3]
                        l23y = olap[abs(dy2 - dy3)]
                        ty   = olap[max(dy2, dy3)]           # overlap of all 3 along y

                        for dz3 in range(8):
                            l13z = olap[dz3]
                            l23z = olap[abs(dz2 - dz3)]
                            tz   = olap[max(dz2, dz3)]       # overlap of all 3 along z

                            V3_cur = tx * ty * tz            # volume in all three
                            if V3_cur != V3:
                                continue

                            x13 = l13x * l13y * l13z         # |C1 ∩ C3|
                            x23 = l23x * l23y * l23z         # |C2 ∩ C3|
                            pair_sum = x12 + x13 + x23
                            V2_cur = pair_sum - 3 * V3_cur   # volume in exactly two

                            if V2_cur != V2:
                                continue

                            V1_cur = 3 * VOL - 2 * V2_cur - 3 * V3_cur
                            if V1_cur != V1:
                                continue

                            # found a configuration
                            print("Yes")
                            print(0, 0, 0, dx2, dy2, dz2, dx3, dy3, dz3)
                            return

    print("No")


if __name__ == "__main__":
    main()