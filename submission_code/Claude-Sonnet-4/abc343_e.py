def solve():
    V1, V2, V3 = map(int, input().split())
    
    # Check if solution is theoretically possible
    total_volume = V1 + V2 + V3
    if total_volume > 3 * 343:  # Can't exceed 3 cube volumes
        print("No")
        return
    
    # Try different arrangements
    # Strategy: Try cubes with controlled overlaps
    
    for dx in range(-7, 8):  # Offset between cube 1 and 2 in x
        for dy in range(-7, 8):  # Offset between cube 1 and 2 in y
            for dz in range(-7, 8):  # Offset between cube 1 and 2 in z
                for dx2 in range(-7, 8):  # Offset between cube 1 and 3 in x
                    for dy2 in range(-7, 8):  # Offset between cube 1 and 3 in y
                        for dz2 in range(-7, 8):  # Offset between cube 1 and 3 in z
                            
                            # Place cubes
                            a1, b1, c1 = 0, 0, 0
                            a2, b2, c2 = dx, dy, dz
                            a3, b3, c3 = dx2, dy2, dz2
                            
                            # Check bounds
                            if any(abs(x) > 100 for x in [a1, b1, c1, a2, b2, c2, a3, b3, c3]):
                                continue
                            
                            # Calculate intersection volumes
                            def intersection_volume(ax1, ay1, az1, ax2, ay2, az2):
                                x_overlap = max(0, min(ax1 + 7, ax2 + 7) - max(ax1, ax2))
                                y_overlap = max(0, min(ay1 + 7, ay2 + 7) - max(ay1, ay2))
                                z_overlap = max(0, min(az1 + 7, az2 + 7) - max(az1, az2))
                                return x_overlap * y_overlap * z_overlap
                            
                            def triple_intersection_volume(ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3):
                                x_overlap = max(0, min(ax1 + 7, ax2 + 7, ax3 + 7) - max(ax1, ax2, ax3))
                                y_overlap = max(0, min(ay1 + 7, ay2 + 7, ay3 + 7) - max(ay1, ay2, ay3))
                                z_overlap = max(0, min(az1 + 7, az2 + 7, az3 + 7) - max(az1, az2, az3))
                                return x_overlap * y_overlap * z_overlap
                            
                            I12 = intersection_volume(a1, b1, c1, a2, b2, c2)
                            I13 = intersection_volume(a1, b1, c1, a3, b3, c3)
                            I23 = intersection_volume(a2, b2, c2, a3, b3, c3)
                            I123 = triple_intersection_volume(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                            
                            # Calculate actual volumes
                            actual_V3 = I123
                            actual_V2 = I12 + I13 + I23 - 3 * I123
                            actual_V1 = 3 * 343 - 2 * (I12 + I13 + I23) + 3 * I123
                            
                            if actual_V1 == V1 and actual_V2 == V2 and actual_V3 == V3:
                                print("Yes")
                                print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                return
    
    print("No")

solve()