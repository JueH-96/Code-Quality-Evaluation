import sys

def solve():
    """
    Solves the three cubes placement problem.
    """
    S = 7
    try:
        V1, V2, V3 = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return

    # From the principle of inclusion-exclusion, a necessary condition is:
    # V1 + 2*V2 + 3*V3 must equal the sum of the volumes of the three cubes.
    if V1 + 2 * V2 + 3 * V3 != S * S * S * 3:
        print("No")
        return

    # The core idea is to decompose the 3D problem into three 1D problems.
    # We precompute all possible 1D overlap configurations.
    # A configuration is a tuple: (overlap12, overlap13, overlap23, overlap123).
    # For each configuration, we store a pair of coordinates (a2, a3) that generates it.
    overlap_configs_map = {}
    
    # A search range of [-2*S, 2*S] for relative coordinates is sufficient to find a solution if one exists.
    coord_range = range(-2 * S, 2 * S + 1)

    for a2 in coord_range:
        for a3 in coord_range:
            # Relative to cube 1 at origin (coord 0)
            o12 = max(0, S - abs(a2))
            o13 = max(0, S - abs(a3))
            o23 = max(0, S - abs(a2 - a3))
            
            min_coord = min(0, a2, a3)
            max_coord = max(0, a2, a3)
            o123 = max(0, S - (max_coord - min_coord))
            
            config = (o12, o13, o23, o123)
            if config not in overlap_configs_map:
                overlap_configs_map[config] = (a2, a3)

    # We search for three 1D configurations (for x, y, z axes) that
    # combine to produce the target volumes V2 and V3.
    configs = list(overlap_configs_map.keys())
    
    # To optimize the search, group configs by their o123 value.
    configs_by_o123 = [[] for _ in range(S + 1)]
    for t in configs:
        configs_by_o123[t[3]].append(t)

    target_sum_v_pair = V2 + 3 * V3

    for t_x in configs:
        o12_x, o13_x, o23_x, o123_x = t_x
        
        for t_y in configs:
            o12_y, o13_y, o23_y, o123_y = t_y

            v123_xy = o123_x * o123_y
            
            z_candidates = []
            if V3 == 0:
                if v123_xy == 0:
                    z_candidates = configs
                else: # v123_xy > 0
                    z_candidates = configs_by_o123[0]
            else: # V3 > 0
                if v123_xy > 0 and V3 % v123_xy == 0:
                    target_o123_z = V3 // v123_xy
                    if 0 <= target_o123_z <= S:
                        z_candidates = configs_by_o123[target_o123_z]

            for t_z in z_candidates:
                o12_z, o13_z, o23_z, _ = t_z
                
                v12 = o12_x * o12_y * o12_z
                v13 = o13_x * o13_y * o13_z
                v23 = o23_x * o23_y * o23_z

                if v12 + v13 + v23 == target_sum_v_pair:
                    # Found a valid combination. Construct the coordinates.
                    a1, b1, c1 = 0, 0, 0
                    a2, a3 = overlap_configs_map[t_x]
                    b2, b3 = overlap_configs_map[t_y]
                    c2, c3 = overlap_configs_map[t_z]
                    
                    print("Yes")
                    print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                    return
    
    # If loops complete without finding a solution.
    print("No")

if __name__ == "__main__":
    solve()