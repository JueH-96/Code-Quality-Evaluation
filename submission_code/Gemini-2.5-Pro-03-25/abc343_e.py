import sys

# Function to calculate intersection volume of two cubes
# C1 is C(a1, b1, c1), C2 is C(a2, b2, c2)
# Cubes have side length S = 7
def intersection_vol2(a1, b1, c1, a2, b2, c2):
    """
    Calculates the volume of the intersection of two axis-aligned cubes C1 and C2.
    C1 is defined by a1 <= x <= a1+S, b1 <= y <= b1+S, c1 <= z <= c1+S
    C2 is defined by a2 <= x <= a2+S, b2 <= y <= b2+S, c2 <= z <= c2+S
    Side length S is fixed at 7.
    """
    S = 7
    # Calculate the absolute differences in coordinates along each axis
    dx = abs(a1 - a2)
    dy = abs(b1 - b2)
    dz = abs(c1 - c2)
    
    # If the distance between centers along any axis is S or more,
    # the cubes do not overlap (or only touch at the boundary), so intersection volume is 0.
    # Equivalently, if the coordinate difference is S or more, they don't overlap.
    if dx >= S or dy >= S or dz >= S:
        return 0
    
    # Calculate the length of the overlapping interval along each axis
    # The overlap length is S minus the coordinate difference.
    Lx = S - dx
    Ly = S - dy
    Lz = S - dz
    
    # The volume of the intersection (a rectangular prism) is the product of the overlap lengths.
    return Lx * Ly * Lz

# Function to calculate intersection volume of three cubes
# C1=C(a1,b1,c1), C2=C(a2,b2,c2), C3=C(a3,b3,c3)
def intersection_vol3(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    """
    Calculates the volume of the intersection of three axis-aligned cubes C1, C2, and C3.
    Side length S is fixed at 7.
    """
    S = 7
    
    # Calculate the intersection interval along the x-axis
    # The start of the intersection interval is the maximum of the start coordinates.
    max_x_start = max(a1, a2, a3)
    # The end of the intersection interval is the minimum of the end coordinates.
    min_x_end = min(a1 + S, a2 + S, a3 + S)
    # The length of the intersection interval along x-axis. If non-positive, intersection is empty.
    Lx = max(0, min_x_end - max_x_start)
    
    # If the x-intersection interval is empty (length 0), the total intersection volume is 0.
    if Lx == 0: 
        return 0
        
    # Calculate the intersection interval along the y-axis
    max_y_start = max(b1, b2, b3)
    min_y_end = min(b1 + S, b2 + S, b3 + S)
    Ly = max(0, min_y_end - max_y_start)

    # If the y-intersection interval is empty, the total intersection volume is 0.
    if Ly == 0: 
        return 0
        
    # Calculate the intersection interval along the z-axis
    max_z_start = max(c1, c2, c3)
    min_z_end = min(c1 + S, c2 + S, c3 + S)
    Lz = max(0, min_z_end - max_z_start)
    
    # If the z-intersection interval is empty, the total intersection volume is 0.
    # This check is somewhat redundant because the final product handles Lz=0, but kept for clarity.
    if Lz == 0: 
        return 0

    # The volume of the intersection (a rectangular prism) is the product of the intersection lengths along each axis.
    return Lx * Ly * Lz

def solve():
    # Read the target volumes V1, V2, V3 from standard input
    V1, V2, V3 = map(int, sys.stdin.readline().split())

    # Check a necessary condition derived from the Principle of Inclusion-Exclusion applied to volumes.
    # The sum V1 + 2*V2 + 3*V3 must equal the sum of the volumes of the three cubes, which is 3 * S^3.
    # Here S=7, so 3 * S^3 = 3 * 343 = 1029.
    # If this condition is not met, no such configuration of cubes can exist.
    if V1 + 2 * V2 + 3 * V3 != 1029: 
        print("No")
        return

    # Without loss of generality, fix the first cube C1 at the origin (0,0,0).
    # This simplifies the search space as we only need to find the relative positions of C2 and C3.
    a1, b1, c1 = 0, 0, 0
    
    # Define the search range for the coordinates of C2 and C3 relative to C1.
    # Cubes intersect only if the distance between their centers is less than S along each axis.
    # This means relative coordinates relevant for intersections are within (-S, S).
    # We search the integer range [-S, S], i.e., [-7, 7], which covers all cases,
    # including boundary cases where intersection volume becomes zero.
    search_bound = 7
    search_range = range(-search_bound, search_bound + 1) # Integer range from -7 to 7 inclusive

    # Brute-force search over all possible integer coordinates for C2 and C3 within the search range.
    # The total number of states to check is (2*search_bound + 1)^6 = 15^6, approximately 11.4 million.
    # This is computationally feasible within typical time limits.
    
    # Iterate through all possible coordinates (a2, b2, c2) for cube C2
    for a2 in search_range:
        for b2 in search_range:
            for c2 in search_range:
                # Calculate the volume of intersection between C1 and C2.
                # This value (v12) depends only on C1 and C2 coordinates, so it can be precomputed
                # before iterating through C3 coordinates for efficiency.
                v12 = intersection_vol2(a1, b1, c1, a2, b2, c2)
                
                # Iterate through all possible coordinates (a3, b3, c3) for cube C3
                for a3 in search_range:
                    for b3 in search_range:
                        for c3 in search_range:
                            
                            # Calculate the remaining intersection volumes involving C3.
                            v13 = intersection_vol2(a1, b1, c1, a3, b3, c3) # Intersection of C1 and C3
                            v23 = intersection_vol2(a2, b2, c2, a3, b3, c3) # Intersection of C2 and C3
                            v123 = intersection_vol3(a1, b1, c1, a2, b2, c2, a3, b3, c3) # Intersection of C1, C2, and C3

                            # Check if the calculated intersection volumes satisfy the conditions required by V2 and V3.
                            # Condition 1: The volume of the intersection of all three cubes (v123) must equal the target V3.
                            # Condition 2: The total volume of regions contained in exactly two cubes must equal the target V2.
                            # This volume can be expressed in terms of pairwise and triple intersection volumes as:
                            # V2 = (v12 - v123) + (v13 - v123) + (v23 - v123) = v12 + v13 + v23 - 3 * v123.
                            # So we check if `v12 + v13 + v23 - 3 * v123 == V2`.
                            
                            if v123 == V3 and (v12 + v13 + v23 - 3 * v123) == V2:
                                
                                # If both conditions are satisfied, we have found a valid configuration.
                                # The coordinates are (a1,b1,c1), (a2,b2,c2), (a3,b3,c3).
                                # Since (a1,b1,c1)=(0,0,0) and a2,b2,c2,a3,b3,c3 are within [-7, 7],
                                # all coordinates automatically satisfy the constraint |coord| <= 100.
                                # No coordinate shifting or further checks are needed.
                                
                                print("Yes")
                                print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
                                return # Exit the function after finding the first valid solution.

    # If the loops complete without finding any configuration satisfying the conditions,
    # it means no solution exists within the searched space. Based on the analysis, this
    # implies no solution exists overall that meets the problem constraints.
    print("No")

# Execute the solve function to run the program.
solve()