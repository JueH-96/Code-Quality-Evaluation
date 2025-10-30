def compute_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    # Helper function to compute the intersection volume between two cubes
    def intersect_vol(a_i, b_i, c_i, a_j, b_j, c_j):
        dx = max(0, min(a_i + 7, a_j + 7) - max(a_i, a_j))
        dy = max(0, min(b_i + 7, b_j + 7) - max(b_i, b_j))
        dz = max(0, min(c_i + 7, c_j + 7) - max(c_i, c_j))
        return dx * dy * dz

    # Calculate all pairwise intersections
    ab = intersect_vol(a1, b1, c1, a2, b2, c2)
    ac = intersect_vol(a1, b1, c1, a3, b3, c3)
    bc = intersect_vol(a2, b2, c2, a3, b3, c3)
    
    # Calculate the triple intersection
    abc = intersect_vol(a1, b1, c1, a2, b2, c2, a3, b3, c3)
    # To compute triple intersection, we need to calculate the intersection of all three
    # First, find intersection of A and B, then intersect with C
    ab_temp = intersect_vol(a1, b1, c1, a2, b2, c2)
    abc = intersect_vol(a1, b1, c1, a2, b2, c2, a3, b3, c3)
    
    # Calculate the volumes
    # V1 is the sum of individual volumes minus the pairwise overlaps plus the triple overlap
    V1 = (343 - ab - ac + abc) + (343 - ab - bc + abc) + (343 - ac - bc + abc)
    # V2 is the sum of pairwise intersections minus 3 times the triple intersection
    V2 = (ab + ac + bc) - 3 * abc
    V3 = abc
    
    return V1, V2, V3

def main():
    import sys
    V1, V2, V3 = map(int, sys.stdin.readline().split())
    
    # Check the necessary condition
    if V1 + 2*V2 + 3*V3 != 1029:
        print("No")
        return
    
    # Try to find a configuration
    # We'll try a limited set of configurations for simplicity
    # Try placing cube1 at (0, 0, 0)
    # Try cube2 along x-axis, varying a2
    # Try cube3 in various positions
    # This is a heuristic and may not cover all cases, but works for some.
    
    # Iterate over possible positions for cube2 and cube3
    for a2 in range(-7, 15):  # a2 can be from -7 to 14 to allow overlaps
        for b2 in range(-7, 15):
            for c2 in range(-7, 15):
                for a3 in range(-7, 15):
                    for b3 in range(-7, 15):
                        for c3 in range(-7, 15):
                            # Compute the volumes
                            V1_cal, V2_cal, V3_cal = compute_volumes(0, 0, 0, a2, b2, c2, a3, b3, c3)
                            if V1_cal == V1 and V2_cal == V2 and V3_cal == V3:
                                print("Yes")
                                print(0, 0, 0, a2, b2, c2, a3, b3, c3)
                                return
    
    # If no configuration found
    print("No")

if __name__ == "__main__":
    main()