import sys

def solve():
    N = int(sys.stdin.readline())
    buildings = []
    for _ in range(N):
        X, H = map(int, sys.stdin.readline().split())
        buildings.append((X, H))

    if N == 1: # Only one building, always visible from (0,0) if X_1>0.
                # Problem implies we need to see ALL buildings. If only one, it is all.
                # It is visible, so max height to NOT see all is not applicable.
                # It's visible from h=0 so print -1.
        print("-1")
        return

    hull_points = [] 
    max_h_val = -float('inf') # Using a very small number; can be negative

    for k_idx in range(N):
        X_k, H_k = buildings[k_idx]

        if len(hull_points) > 0:
            # Binary search for the tangent point P_j from P_k to the hull
            # P_j = hull_points[idx] that minimizes slope (H_k - P_j.H) / (X_k - P_j.X)
            
            search_l, search_r = 0, len(hull_points) - 1
            best_j_idx_for_k = search_l # Default to first point
            
            # This binary search finds the leftmost index `idx` in `[search_l, search_r]`
            # such that `slope(hull_points[idx], P_k)` is minimal.
            # It works because slope(hull_points[i], P_k) is V-shaped.
            # Loop while search_l < search_r.
            # temp_l will be the index of the minimum.
            temp_l, temp_r = search_l, search_r
            while temp_l < temp_r:
                mid = temp_l + (temp_r - temp_l) // 2
                # Compare slope(hull_points[mid], P_k) vs slope(hull_points[mid+1], P_k)
                # Let P_mid = hull_points[mid], P_mid1 = hull_points[mid+1]
                # slope_mid < slope_mid1  is equivalent to
                # (H_k - P_mid.H)*(X_k - P_mid1.X) < (H_k - P_mid1.H)*(X_k - P_mid.X)
                # because X_k - P_mid.X > 0 and X_k - P_mid1.X > 0 (as X_k > X_on_hull)
                
                # Numerators could be 0, but denominators are X_k - X_j > 0
                # (H_k - hull_points[mid][1]) / (X_k - hull_points[mid][0])
                # vs
                # (H_k - hull_points[mid+1][1]) / (X_k - hull_points[mid+1][0])

                # Using cross-product form to keep it integer arithmetic
                # (H_k - H_mid) * (X_k - X_mid1) vs (H_k - H_mid1) * (X_k - X_mid)
                val1_num = H_k - hull_points[mid][1]
                val1_den = X_k - hull_points[mid][0]
                val2_num = H_k - hull_points[mid+1][1]
                val2_den = X_k - hull_points[mid+1][0]
                
                # Compare val1_num/val1_den vs val2_num/val2_den
                # Equivalent to val1_num * val2_den vs val2_num * val1_den
                # Denominators are positive.
                if val1_num * val2_den < val2_num * val1_den: # slope at mid is less than slope at mid+1
                    temp_r = mid # minimum is in [temp_l, mid]
                else: # slope at mid is >= slope at mid+1
                    temp_l = mid + 1 # minimum is in [mid+1, temp_r]
            best_j_idx_for_k = temp_l
            
            X_j, H_j = hull_points[best_j_idx_for_k]
            
            current_h = (H_j * X_k - H_k * X_j) / (X_k - X_j)
            if current_h > max_h_val:
                max_h_val = current_h

        # Add P_k to hull_points, maintaining upper convex hull property
        # Pop points P_b from hull if P_a, P_b, P_k don't make a "strict left turn"
        # (i.e., P_a, P_b, P_k make a right turn or are collinear).
        # This means slope(P_a,P_b) <= slope(P_b,P_k).
        while len(hull_points) >= 2:
            P_b = hull_points[-1] # (X_b, H_b)
            P_a = hull_points[-2] # (X_a, H_a)
            
            # We pop P_b if slope(P_a,P_b) <= slope(P_b,P_k)
            # (P_b[1]-P_a[1]) / (P_b[0]-P_a[0]) <= (H_k-P_b[1]) / (X_k-P_b[0])
            # Since denominators (P_b[0]-P_a[0]) > 0 and (X_k-P_b[0]) > 0:
            # (P_b[1]-P_a[1]) * (X_k-P_b[0]) <= (H_k-P_b[1]) * (P_b[0]-P_a[0])
            
            lhs_val = (P_b[1]-P_a[1]) * (X_k-P_b[0])
            rhs_val = (H_k-P_b[1]) * (P_b[0]-P_a[0])

            if lhs_val <= rhs_val:
                hull_points.pop()
            else:
                break
        hull_points.append((X_k, H_k))

    if max_h_val < 0: # If all h_jk < 0, implies all buildings visible from (0,0)
        print("-1")
    else:
        print(f"{max_h_val:.17f}")

solve()