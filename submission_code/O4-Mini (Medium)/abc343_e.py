import sys
import threading
def main():
    import sys

    V1, V2, V3 = map(int, sys.stdin.readline().split())
    # total single cube volume = 343, sum_i vol(Ci)=3*343=1029
    # consistency: V1 == 1029 - 2*V2 - 3*V3
    if V1 != 1029 - 2*V2 - 3*V3:
        print("No")
        return

    # helper for one-dimensional overlap length
    def ov(a1, a2):
        # Cube1 spans [0,7], Cube2 spans [a2, a2+7]
        left = max(0, a2)
        right = min(7, a2+7)
        return max(0, right - left)

    found = False
    # try all small integer offsets for C2 and C3 relative to C1=(0,0,0)
    for dx2 in range(0, 8):
        if found: break
        for dy2 in range(0, 8):
            if found: break
            for dz2 in range(0, 8):
                # compute C1∩C2
                x12 = ov(0, dx2)
                y12 = ov(0, dy2)
                z12 = ov(0, dz2)
                I12 = x12 * y12 * z12
                for dx3 in range(0, 8):
                    if found: break
                    for dy3 in range(0, 8):
                        if found: break
                        for dz3 in range(0, 8):
                            # compute pair intersections
                            x13 = ov(0, dx3)
                            y13 = ov(0, dy3)
                            z13 = ov(0, dz3)
                            I13 = x13 * y13 * z13

                            # C2 vs C3
                            # spans C2: [dx2, dx2+7], C3: [dx3, dx3+7]
                            def ov23(a2, a3):
                                left = max(a2, a3)
                                right = min(a2+7, a3+7)
                                return max(0, right - left)

                            x23 = ov23(dx2, dx3)
                            y23 = ov23(dy2, dy3)
                            z23 = ov23(dz2, dz3)
                            I23 = x23 * y23 * z23

                            # triple intersection
                            # along each axis
                            def ov3(a2, a3):
                                left = max(0, a2, a3)
                                right = min(7, a2+7, a3+7)
                                return max(0, right - left)
                            x123 = ov3(dx2, dx3)
                            y123 = ov3(dy2, dy3)
                            z123 = ov3(dz2, dz3)
                            T = x123 * y123 * z123

                            # compute exact-two region volume and exact-one
                            V3_calc = T
                            V2_calc = I12 + I13 + I23 - 3 * T
                            # we need not recompute V1 exactly, consistency already checked
                            if V3_calc == V3 and V2_calc == V2:
                                # found
                                print("Yes")
                                # C1 at (0,0,0), C2 at (dx2,dy2,dz2), C3 at (dx3,dy3,dz3)
                                print(0, 0, 0, dx2, dy2, dz2, dx3, dy3, dz3)
                                found = True
                                break
    if not found:
        print("No")

if __name__ == "__main__":
    main()