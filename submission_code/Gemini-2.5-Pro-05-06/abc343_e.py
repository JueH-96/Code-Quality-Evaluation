import sys

def solve():
    V1_target, V2_target, V3_target = map(int, sys.stdin.readline().split())

    S = 7
    S_cubed = S * S * S

    if V1_target + 2 * V2_target + 3 * V3_target != 3 * S_cubed:
        print("No")
        return
    
    # Cube 1 is fixed at (0,0,0)
    a1, b1, c1 = 0, 0, 0

    # Iterate coordinates for C2 and C3 with components in [0, S]
    # This gives (S+1)^6 states. For S=7, this is 8^6 = 262,144 iterations.
    # The coordinates will be small, e.g. in [0,7], satisfying |coord| <= 100.
    for a2 in range(S + 1):
        for b2 in range(S + 1):
            for c2 in range(S + 1):
                for a3 in range(S + 1):
                    for b3 in range(S + 1):
                        for c3 in range(S + 1):
                            # Calculate overlap lengths.
                            # Given a1=0 and a2,a3 are in [0,S]:
                            # X dimension overlaps
                            x12 = S - a2  # S - abs(a1-a2) = S - abs(0-a2) = S - a2 (since a2 >= 0)
                                          # max(0, S-a2) is S-a2 (since a2 <= S means S-a2 >= 0)
                            x13 = S - a3
                            x23 = max(0, S - abs(a2 - a3)) # abs needed as a2,a3 order unknown
                            # x123 for a1=0, a2>=0, a3>=0: max(0, min(S, a2+S, a3+S) - max(0,a2,a3))
                            #      = max(0, S - max(a2,a3)) (since a2,a3 >= 0 => a_k+S >= S)
                            #      = S - max(a2,a3) (since a2,a3 <= S => max(a2,a3) <= S => S-max(a2,a3) >=0)
                            x123 = S - max(a2, a3)

                            # Y dimension overlaps (similarly)
                            y12 = S - b2
                            y13 = S - b3
                            y23 = max(0, S - abs(b2 - b3))
                            y123 = S - max(b2, b3)

                            # Z dimension overlaps (similarly)
                            z12 = S - c2
                            z13 = S - c3
                            z23 = max(0, S - abs(c2 - c3))
                            z123 = S - max(c2, c3)
                            
                            # Volume of intersection of all three cubes (C1 n C2 n C3)
                            vol123_calc = x123 * y123 * z123
                            if vol123_calc != V3_target:
                                continue

                            # Volumes of pairwise intersections
                            vol12_calc = x12 * y12 * z12    # Vol(C1 n C2)
                            vol13_calc = x13 * y13 * z13    # Vol(C1 n C3)
                            vol23_calc = x23 * y23 * z23    # Vol(C2 n C3)
                            
                            # Volume of regions in exactly two cubes
                            # V2 = sum over pairs (i,j) of ( Vol(Ci n Cj) - Vol(C1 n C2 n C3) )
                            vol_in_exactly_two_calc = (vol12_calc - vol123_calc) + \
                                                      (vol13_calc - vol123_calc) + \
                                                      (vol23_calc - vol123_calc)
                            
                            if vol_in_exactly_two_calc != V2_target:
                                continue
                            
                            # If V3 and V2 targets are met, V1 target is also met due to the
                            # initial check V1 + 2*V2 + 3*V3 = 3*S_cubed.
                            print("Yes")
                            # Output format: a1 b1 c1 a2 b2 c2 a3 b3 c3
                            print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                            return
    
    # If loops complete, no solution found in the searched range.
    print("No")

solve()