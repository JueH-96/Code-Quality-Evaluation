import math

def solve():
    V1, V2, V3 = map(int, input().split())

    L = 7
    L_cubed = L**3
    total_volume_cubes = 3 * L_cubed # Sum of volumes of C1, C2, C3

    # Initial consistency check based on Principle of Inclusion-Exclusion
    # V1 + 2*V2 + 3*V3 must equal the sum of volumes of the three cubes (3*L^3)
    # This relation is derived from:
    # V_total = V1 + V2 + V3
    # Sum of volumes = V_total + V2 + 2*V3 = 3*L^3
    # V1 + V2 + V3 + V2 + 2*V3 = V1 + 2*V2 + 3*V3
    if V1 + 2 * V2 + 3 * V3 != total_volume_cubes:
        print("No")
        return

    # We try a specific geometric configuration for the cubes:
    # C1 at (0,0,0)
    # C2 at (0, k_y, 0)
    # C3 at (k_x, 0, 0)
    # All cubes share the same z-interval [0,7], so z-overlap is always L=7.
    # k_x and k_y define the shift, and thus the overlap in x and y.
    # The overlap length in a dimension 'd' for intervals [s1, s1+L] and [s2, s2+L] is max(0, L - |s1-s2|).
    # Let ov_x = L - k_x and ov_y = L - k_y be the effective overlap lengths in x and y dimensions.
    # We require 0 <= k_x <= L and 0 <= k_y <= L for positive overlaps, which means 0 <= ov_x <= L and 0 <= ov_y <= L.
    # All coordinates will be between 0 and L (i.e., 0 to 7), satisfying the <= 100 constraint.

    # Based on this setup:
    # V3 (volume in exactly three cubes) = Y = ov_x * ov_y * L
    # V2 (volume in exactly two cubes) = L * (L * (ov_x + ov_y) - 2 * ov_x * ov_y)

    # Case 1: V3 = 0
    if V3 == 0:
        # If V3 = 0, then ov_x * ov_y * L = 0, which implies ov_x = 0 or ov_y = 0 (or both).
        
        # Subcase 1.1: ov_x = 0 (i.e., k_x = L)
        # In this case, C1 and C3 only touch at a face or don't overlap in x,
        # leading to C1_x_overlap_C3_x = 0.
        # So X13 = 0 and Y = 0.
        # The formula for V2 simplifies: V2 = L * (L * (0 + ov_y) - 2 * 0 * ov_y) = L^2 * ov_y.
        # We check if V2 is a multiple of L^2 and if the resulting ov_y is valid.
        if V2 % (L * L) == 0:
            current_ov_y = V2 // (L * L)
            if 0 <= current_ov_y <= L:
                k_x = L  # ov_x = 0 implies k_x = L
                k_y = L - current_ov_y # ov_y = current_ov_y implies k_y = L - current_ov_y
                print("Yes")
                print(f"0 0 0 0 {k_y} 0 {k_x} 0 0")
                return # Found a solution, exit

        # Subcase 1.2: ov_y = 0 (i.e., k_y = L)
        # Symmetric to Subcase 1.1. Only check if a solution hasn't been found yet.
        # This covers cases where only C1 and C3 overlap partially, or where nothing overlaps.
        # (e.g., if V2=0, then ov_x=0 for subcase 1.1 will result in ov_y=0, k_y=L.
        # The result (0,7,0) and (7,0,0) will be printed by subcase 1.1, so we don't need to re-print).
        if V2 % (L * L) == 0: # This condition might have been met by Subcase 1.1 already if V2=0.
            current_ov_x = V2 // (L * L)
            if 0 <= current_ov_x <= L:
                k_x = L - current_ov_x # ov_x = current_ov_x implies k_x = L - current_ov_x
                k_y = L # ov_y = 0 implies k_y = L
                print("Yes")
                print(f"0 0 0 0 {k_y} 0 {k_x} 0 0")
                return # Found a solution, exit

    # Case 2: V3 > 0
    else:
        # If V3 > 0, then ov_x and ov_y must both be positive integers.
        # From V3 = ov_x * ov_y * L, V3 must be divisible by L.
        if V3 % L == 0:
            prod_ov = V3 // L # This is the product ov_x * ov_y
            
            # Iterate through possible integer values for ov_x from 1 to L (inclusive).
            # If ov_x is a divisor of prod_ov, then ov_y can be determined.
            for current_ov_x in range(1, L + 1):
                if prod_ov % current_ov_x == 0:
                    current_ov_y = prod_ov // current_ov_x
                    
                    # Check if current_ov_y is within the valid range [1, L]
                    if 1 <= current_ov_y <= L:
                        # Calculate V2_calculated using the derived ov_x and ov_y
                        V2_calculated = L * (L * (current_ov_x + current_ov_y) - 2 * current_ov_x * current_ov_y)
                        
                        # If the calculated V2 matches the input V2, we found a solution.
                        if V2_calculated == V2:
                            k_x = L - current_ov_x
                            k_y = L - current_ov_y
                            
                            print("Yes")
                            print(f"0 0 0 0 {k_y} 0 {k_x} 0 0")
                            return # Found a solution, exit

    # If no solution was found by the specific configuration, print "No".
    print("No")

solve()