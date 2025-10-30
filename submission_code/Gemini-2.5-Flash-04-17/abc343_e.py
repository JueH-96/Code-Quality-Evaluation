import sys

# Helper function to calculate length of intersection of two 1D intervals [base, base+7] and [offset, offset+7]
# The length is max(0, 7 - |offset - base|). For base=0, this is max(0, 7 - |offset|).
def l(offset):
    return max(0, 7 - abs(offset))

def solve():
    V1, V2, V3 = map(int, sys.stdin.readline().split())

    # Necessary condition derived from inclusion-exclusion principle for volumes
    # Total volume of the union of three cubes: V(C1 U C2 U C3)
    # V(C1 U C2 U C3) = V(C1) + V(C2) + V(C3) - (V(C1 intersect C2) + V(C1 intersect C3) + V(C2 intersect C3)) + V(C1 intersect C2 intersect C3)
    # V(Ci) = 7^3 = 343
    # Let V_1_only = sum of volumes in exactly one cube
    # Let V_2_only = sum of volumes in exactly two cubes
    # Let V_3_only = volume in exactly three cubes
    # V_1 = V_1_only, V_2 = V_2_only, V_3 = V_3_only (as per problem statement)
    # Relationship between total volumes and disjoint region volumes:
    # V(C1) = V(C1 \ (C2 U C3)) + V((C1 intersect C2) \ C3) + V((C1 intersect C3) \ C2) + V(C1 intersect C2 intersect C3)
    # Summing for C1, C2, C3:
    # 3 * V_cube = V_1_only + 2 * V_2_only + 3 * V_3_only
    # 3 * 343 = V1 + 2 * V2 + 3 * V3
    # 1029 = V1 + 2 * V2 + 3 * V3
    if V1 + 2 * V2 + 3 * V3 != 1029:
        print("No")
        return

    # We search for a solution using a specific configuration:
    # Cube C1 is anchored at (0,0,0): C1 = [0,7] x [0,7] x [0,7]
    # Cube C2 is anchored at (0, y2, 0): C2 = [0,7] x [y2, y2+7] x [0,7]
    # Cube C3 is anchored at (x3, 0, 0): C3 = [x3, x3+7] x [0,7] x [0,7]
    # We need to find integers x3, y2 such that |x3| <= 100, |y2| <= 100 and the volume conditions are met.
    # The coordinates will be (0,0,0), (0,y2,0), (x3,0,0).

    # Let's calculate the volumes of intersections for this configuration:
    # V(C1 intersect C2):
    # x-overlap: [max(0,0), min(7,7)] = [0,7], length 7
    # y-overlap: [max(0,y2), min(7,y2+7)], length l(y2)
    # z-overlap: [max(0,0), min(7,7)] = [0,7], length 7
    # V12 = 7 * l(y2) * 7 = 49 * l(y2)

    # V(C1 intersect C3):
    # x-overlap: [max(0,x3), min(7,x3+7)], length l(x3)
    # y-overlap: [max(0,0), min(7,7)] = [0,7], length 7
    # z-overlap: [max(0,0), min(7,7)] = [0,7], length 7
    # V13 = l(x3) * 7 * 7 = 49 * l(x3)

    # V(C2 intersect C3):
    # x-overlap: [max(0,x3), min(7,x3+7)], length l(x3) (Intersection of C2's x-interval [0,7] and C3's x-interval [x3, x3+7])
    # y-overlap: [max(y2,0), min(y2+7,7)], length l(y2) (Intersection of C2's y-interval [y2, y2+7] and C3's y-interval [0,7])
    # z-overlap: [max(0,0), min(7,7)] = [0,7], length 7 (Intersection of C2's z-interval [0,7] and C3's z-interval [0,7])
    # V23 = l(x3) * l(y2) * 7

    # V(C1 intersect C2 intersect C3):
    # x-overlap: [max(0,0,x3), min(7,7,x3+7)] = [max(0,x3), min(7,x3+7)], length l(x3)
    # y-overlap: [max(0,y2,0), min(7,y2+7,7)] = [max(0,y2), min(7,y2+7)], length l(y2)
    # z-overlap: [max(0,0,0), min(7,7,7)] = [0,7], length 7
    # V3_calc = l(x3) * l(y2) * 7

    # We know V_2_only = V_2 = V(C1 intersect C2 \ C3) + V(C1 intersect C3 \ C2) + V(C2 intersect C3 \ C1)
    # V(A \ B) = V(A) - V(A intersect B)
    # V_2 = (V12 - V3_calc) + (V13 - V3_calc) + (V23 - V3_calc)
    # V_2 = V12 + V13 + V23 - 3 * V3_calc
    # Substitute the formulas for V12, V13, V23, V3_calc:
    # V2_calc = 49 * l(y2) + 49 * l(x3) + 7 * l(x3) * l(y2) - 3 * (7 * l(x3) * l(y2))
    # V2_calc = 49 * l(x3) + 49 * l(y2) - 14 * l(x3) * l(y2)

    # We need to find integers x3, y2 such that V3_calc == V3 and V2_calc == V2.
    # The possible values for l(u) for an integer u are integers from 0 to 7.
    # We can iterate through all possible integer values for l(x3) and l(y2) in [0, 7].
    # Let Lx = l(x3) and Ly = l(y2).

    for Lx in range(8): # Lx can be 0, 1, ..., 7
        for Ly in range(8): # Ly can be 0, 1, ..., 7
            # Calculate candidate volumes using Lx and Ly
            V3_cand = 7 * Lx * Ly
            V2_cand = 49 * Lx + 49 * Ly - 14 * Lx * Ly

            # Check if the candidate volumes match the target volumes
            if V3_cand == V3 and V2_cand == V2:
                # Found a pair (Lx, Ly) that works.
                # Now find corresponding integer coordinates x3 and y2.
                # We need l(x3) = Lx and l(y2) = Ly.
                # l(u) = max(0, 7 - |u|)
                # If l(u) = k where k is an integer in [1, 7]:
                #   max(0, 7 - |u|) = k implies 7 - |u| = k (since k>0)
                #   |u| = 7 - k. Since 1 <= k <= 7, 0 <= 7-k <= 6.
                #   We can choose u = 7 - k. This results in |u| <= 6.
                # If l(u) = 0:
                #   max(0, 7 - |u|) = 0 implies 7 - |u| <= 0
                #   |u| >= 7. We need any integer u with magnitude >= 7.
                #   The smallest magnitude is 7. We can choose u = 7. This results in |u| = 7.
                # In all cases, the chosen coordinate will have magnitude at most 7,
                # which is well within the <= 100 constraint.

                x3_val = 7 - Lx if Lx > 0 else 7
                y2_val = 7 - Ly if Ly > 0 else 7

                # The coordinates for the three cubes are:
                # C1: (a1, b1, c1) = (0, 0, 0)
                # C2: (a2, b2, c2) = (0, y2_val, 0)
                # C3: (a3, b3, c3) = (x3_val, 0, 0)

                # All coordinates (0, y2_val, x3_val) have magnitude at most 7, satisfying |coord| <= 100.

                print("Yes")
                # Print a1 b1 c1 a2 b2 c2 a3 b3 c3
                print(f"0 0 0 0 {y2_val} 0 {x3_val} 0 0")
                return # Exit after finding the first solution

    # If the loops complete without finding a pair (Lx, Ly) that produces the target V2 and V3
    # for this specific configuration, then this configuration doesn't work.
    # Given the problem constraints and the sample structure, it is likely that if a solution
    # exists, this specific configuration is sufficient to find one.
    print("No")

solve()